import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
import argparse
import logging
from tqdm import tqdm
import random

# Import the model from app.py
from app import PartialConvUNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InpaintingDataset(Dataset):
    """Dataset for inpainting training"""
    
    def __init__(self, image_dir, mask_dir=None, image_size=256, create_random_masks=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.create_random_masks = create_random_masks
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if mask_dir and os.path.exists(mask_dir):
            self.mask_files = [f for f in os.listdir(mask_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.mask_files = []
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        logger.info(f"Dataset initialized with {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def create_random_mask(self, size):
        """Create random irregular mask"""
        mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Create random shapes
        num_shapes = random.randint(1, 5)
        for _ in range(num_shapes):
            # Random rectangles
            if random.random() < 0.5:
                x1, y1 = random.randint(0, size//2), random.randint(0, size//2)
                x2, y2 = random.randint(size//2, size), random.randint(size//2, size)
                mask[y1:y2, x1:x2] = 0
            
            # Random circles
            else:
                center = (random.randint(size//4, 3*size//4), random.randint(size//4, 3*size//4))
                radius = random.randint(size//8, size//4)
                cv2.circle(mask, center, radius, 0, -1)
        
        # Add some noise
        if random.random() < 0.3:
            noise = np.random.random((size, size)) < 0.1
            mask[noise] = 0
        
        return mask
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load or create mask
        if self.mask_files and idx < len(self.mask_files):
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
        else:
            mask = self.create_random_mask(self.image_size)
        
        # Convert mask to PIL Image
        mask_pil = Image.fromarray(mask)
        
        # Apply transforms
        image_tensor = self.transform(image)
        mask_tensor = self.mask_transform(mask_pil)
        
        # Invert mask (1 for known pixels, 0 for holes)
        mask_tensor = 1 - (mask_tensor > 0.5).float()
        
        # Create ground truth (original image)
        gt_tensor = image_tensor.clone()
        
        # Create input (masked image)
        input_tensor = image_tensor * mask_tensor
        
        return {
            'image': input_tensor,
            'mask': mask_tensor,
            'gt': gt_tensor
        }

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use a simple L1 loss for now (can be enhanced with VGG features)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, input, target, mask):
        # Hole loss (loss in masked regions)
        hole_loss = self.l1_loss(input * (1 - mask), target * (1 - mask))
        
        # Valid loss (loss in non-masked regions)
        valid_loss = self.l1_loss(input * mask, target * mask)
        
        # Combine losses
        total_loss = hole_loss * 6.0 + valid_loss * 1.0
        
        return total_loss, hole_loss, valid_loss

def train_model(args):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = InpaintingDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        image_size=args.image_size,
        create_random_masks=args.random_masks
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = PartialConvUNet(input_channels=3, output_channels=3)
    model.to(device)
    
    # Loss function and optimizer
    criterion = PerceptualLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=0.1)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_hole_loss = 0.0
        epoch_valid_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['gt'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, masks)
            
            # Calculate loss
            total_loss, hole_loss, valid_loss = criterion(outputs, targets, masks)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_hole_loss += hole_loss.item()
            epoch_valid_loss += valid_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Hole': f'{hole_loss.item():.4f}',
                'Valid': f'{valid_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_hole_loss = epoch_hole_loss / len(train_loader)
        avg_valid_loss = epoch_valid_loss / len(train_loader)
        
        logger.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Hole={avg_hole_loss:.4f}, Valid={avg_valid_loss:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(args.output_dir, 'partial_conv_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f'Saved best model with loss: {best_loss:.4f}')
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    logger.info('Training completed!')

def download_sample_datasets():
    """Download sample datasets for training"""
    logger.info("This function would download Places2 and CelebA datasets.")
    logger.info("For now, please manually download datasets and place them in the appropriate directories.")
    logger.info("Places2: http://places2.csail.mit.edu/download.html")
    logger.info("CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")

def main():
    parser = argparse.ArgumentParser(description='Train Partial Convolution model for inpainting')
    
    parser.add_argument('--train_images', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--train_masks', type=str, default=None,
                        help='Path to training masks directory (optional)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for saved models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of input images')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--decay_steps', type=int, default=50,
                        help='Decay learning rate every N steps')
    parser.add_argument('--random_masks', action='store_true',
                        help='Create random masks if mask directory not provided')
    parser.add_argument('--download_datasets', action='store_true',
                        help='Download sample datasets')
    
    args = parser.parse_args()
    
    if args.download_datasets:
        download_sample_datasets()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_model(args)

if __name__ == '__main__':
    main()
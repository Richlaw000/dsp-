from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

GALLERY_FOLDER = 'static/gallery'
MODEL_FOLDER = 'models'
os.makedirs(GALLERY_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Partial Convolution Layer Implementation
class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(PartialConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # Initialize mask conv weights to 1
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        
        # Register mask conv as non-trainable
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input_x, mask):
        # Convolution on input
        output = self.input_conv(input_x * mask)
        
        # Convolution on mask to get normalization factor
        if self.input_conv.bias is not None:
            bias_view = self.input_conv.bias.view(1, -1, 1, 1)
        else:
            bias_view = None
            
        # Calculate sum of mask weights for normalization
        sum_mask = self.mask_conv(mask)
        
        # Avoid division by zero
        no_update_holes = sum_mask == 0
        sum_mask[no_update_holes] = 1.0
        
        # Normalize output
        output = output / sum_mask
        
        if bias_view is not None:
            output = output + bias_view
            
        # Update mask
        new_mask = torch.ones_like(sum_mask)
        new_mask[no_update_holes] = 0.0
        
        return output, new_mask

# Partial Convolution UNet Model
class PartialConvUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(PartialConvUNet, self).__init__()
        
        # Encoder
        self.enc1 = PartialConv2d(input_channels, 64, 7, stride=2, padding=3)
        self.enc2 = PartialConv2d(64, 128, 5, stride=2, padding=2)
        self.enc3 = PartialConv2d(128, 256, 5, stride=2, padding=2)
        self.enc4 = PartialConv2d(256, 512, 3, stride=2, padding=1)
        self.enc5 = PartialConv2d(512, 512, 3, stride=2, padding=1)
        
        # Decoder
        self.dec5 = PartialConv2d(512 + 512, 512, 3, padding=1)
        self.dec4 = PartialConv2d(512 + 256, 256, 3, padding=1)
        self.dec3 = PartialConv2d(256 + 128, 128, 3, padding=1)
        self.dec2 = PartialConv2d(128 + 64, 64, 3, padding=1)
        self.dec1 = PartialConv2d(64 + input_channels, output_channels, 3, padding=1)
        
        # Batch normalization layers
        self.bn_enc1 = nn.BatchNorm2d(64)
        self.bn_enc2 = nn.BatchNorm2d(128)
        self.bn_enc3 = nn.BatchNorm2d(256)
        self.bn_enc4 = nn.BatchNorm2d(512)
        self.bn_enc5 = nn.BatchNorm2d(512)
        
        self.bn_dec5 = nn.BatchNorm2d(512)
        self.bn_dec4 = nn.BatchNorm2d(256)
        self.bn_dec3 = nn.BatchNorm2d(128)
        self.bn_dec2 = nn.BatchNorm2d(64)

    def forward(self, input_x, mask):
        # Encoder
        enc1, mask1 = self.enc1(input_x, mask)
        enc1 = F.relu(self.bn_enc1(enc1))
        
        enc2, mask2 = self.enc2(enc1, mask1)
        enc2 = F.relu(self.bn_enc2(enc2))
        
        enc3, mask3 = self.enc3(enc2, mask2)
        enc3 = F.relu(self.bn_enc3(enc3))
        
        enc4, mask4 = self.enc4(enc3, mask3)
        enc4 = F.relu(self.bn_enc4(enc4))
        
        enc5, mask5 = self.enc5(enc4, mask4)
        enc5 = F.relu(self.bn_enc5(enc5))
        
        # Decoder with skip connections
        dec5_input = torch.cat([enc5, enc4], dim=1)
        mask5_up = F.interpolate(mask5, size=mask4.shape[2:], mode='nearest')
        dec5, mask_dec5 = self.dec5(dec5_input, mask5_up)
        dec5 = F.relu(self.bn_dec5(dec5))
        
        dec4_input = torch.cat([dec5, enc3], dim=1)
        mask4_up = F.interpolate(mask_dec5, size=mask3.shape[2:], mode='nearest')
        dec4, mask_dec4 = self.dec4(dec4_input, mask4_up)
        dec4 = F.relu(self.bn_dec4(dec4))
        
        dec3_input = torch.cat([dec4, enc2], dim=1)
        mask3_up = F.interpolate(mask_dec4, size=mask2.shape[2:], mode='nearest')
        dec3, mask_dec3 = self.dec3(dec3_input, mask3_up)
        dec3 = F.relu(self.bn_dec3(dec3))
        
        dec2_input = torch.cat([dec3, enc1], dim=1)
        mask2_up = F.interpolate(mask_dec3, size=mask1.shape[2:], mode='nearest')
        dec2, mask_dec2 = self.dec2(dec2_input, mask2_up)
        dec2 = F.relu(self.bn_dec2(dec2))
        
        dec1_input = torch.cat([dec2, input_x], dim=1)
        mask1_up = F.interpolate(mask_dec2, size=mask.shape[2:], mode='nearest')
        output, _ = self.dec1(dec1_input, mask1_up)
        
        return torch.tanh(output)

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the Partial Convolution model"""
    global model
    try:
        model = PartialConvUNet(input_channels=3, output_channels=3)
        model_path = os.path.join(MODEL_FOLDER, 'partial_conv_model.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Loaded pretrained Partial Convolution model")
        else:
            logger.info("No pretrained model found. Using randomly initialized weights.")
            
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Image Generators (existing code)
def generate_checkerboard(size=256, num_squares=8):
    square_size = size // num_squares
    checkerboard = np.zeros((size, size), dtype=np.uint8)
    for row in range(num_squares):
        for col in range(num_squares):
            if (row + col) % 2 == 0:
                checkerboard[
                    row * square_size:(row + 1) * square_size,
                    col * square_size:(col + 1) * square_size,
                ] = 255
    return checkerboard

def generate_sinewave(size=256):
    x = np.linspace(0, 4 * np.pi, size)
    y = 127.5 + 127.5 * np.sin(x)
    sinewave_img = np.tile(y, (size, 1)).astype(np.uint8)
    return sinewave_img

def generate_noise(size=256):
    noise_img = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    return noise_img

def save_generated_images():
    cv2.imwrite(os.path.join(GALLERY_FOLDER, 'checkerboard.png'), generate_checkerboard())
    cv2.imwrite(os.path.join(GALLERY_FOLDER, 'sinewave.png'), generate_sinewave())
    cv2.imwrite(os.path.join(GALLERY_FOLDER, 'noise.png'), generate_noise())

# Classical inpainting using OpenCV
def classical_inpainting(image, mask, method='telea'):
    """Perform classical inpainting using OpenCV"""
    if method == 'telea':
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    else:  # navier-stokes
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    return inpainted

# Deep learning inpainting using Partial Convolution
def deep_inpainting(image, mask):
    """Perform deep learning inpainting using Partial Convolution model"""
    if model is None:
        raise ValueError("Model not loaded")
    
    # Preprocess image and mask
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert numpy arrays to PIL Images
    if len(image.shape) == 3:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = Image.fromarray(image)
    
    mask_pil = Image.fromarray(mask).convert('L')
    
    # Apply transforms
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    mask_tensor = transforms.ToTensor()(mask_pil).unsqueeze(0).to(device)
    
    # Invert mask (model expects 1 for known pixels, 0 for holes)
    mask_tensor = 1 - mask_tensor
    
    with torch.no_grad():
        output = model(image_tensor, mask_tensor)
        
    # Denormalize output
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    output = output * std + mean
    output = torch.clamp(output, 0, 1)
    
    # Convert back to numpy
    output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_np = (output_np * 255).astype(np.uint8)
    
    return cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

# Combined inpainting pipeline
def combined_inpainting(image, mask):
    """Combined classical + deep learning inpainting pipeline"""
    try:
        # Step 1: Classical inpainting as preprocessing
        classical_result = classical_inpainting(image, mask, method='telea')
        
        # Step 2: Deep learning refinement
        if model is not None:
            deep_result = deep_inpainting(classical_result, mask)
            return deep_result
        else:
            logger.warning("Deep learning model not available, using classical result only")
            return classical_result
    except Exception as e:
        logger.error(f"Error in combined inpainting: {e}")
        # Fallback to classical inpainting
        return classical_inpainting(image, mask, method='telea')

# Initialize model and generated images
load_model()
save_generated_images()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gallery')
def gallery():
    images = [f'/static/gallery/{img}' for img in os.listdir(GALLERY_FOLDER) if img.endswith('.png')]
    return render_template('gallery.html', images=images)

@app.route('/edge-detect', methods=['POST'])
def edge_detect():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Sobel edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    
    # Encode images as PNG and base64
    _, gray_encoded = cv2.imencode('.png', gray)
    _, sobel_encoded = cv2.imencode('.png', sobel)
    
    gray_b64 = base64.b64encode(gray_encoded).decode('utf-8')
    sobel_b64 = base64.b64encode(sobel_encoded).decode('utf-8')
    
    return jsonify({"gray": gray_b64, "sobel": sobel_b64})

@app.route('/remove-object', methods=['POST'])
def remove_object():
    """Object removal endpoint using combined inpainting pipeline"""
    try:
        # Get uploaded files
        image_file = request.files.get('image')
        mask_file = request.files.get('mask')
        
        if not image_file or not mask_file:
            return jsonify({"error": "Both image and mask files are required"}), 400
        
        # Decode image
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Decode mask
        mask_bytes = np.frombuffer(mask_file.read(), np.uint8)
        mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return jsonify({"error": "Invalid mask file"}), 400
        
        # Resize to consistent size
        target_size = (256, 256)
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)
        
        # Threshold mask to ensure binary values
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply combined inpainting pipeline
        result = combined_inpainting(image, mask)
        
        # Encode results
        _, original_encoded = cv2.imencode('.png', image)
        _, mask_encoded = cv2.imencode('.png', mask)
        _, result_encoded = cv2.imencode('.png', result)
        
        original_b64 = base64.b64encode(original_encoded).decode('utf-8')
        mask_b64 = base64.b64encode(mask_encoded).decode('utf-8')
        result_b64 = base64.b64encode(result_encoded).decode('utf-8')
        
        return jsonify({
            "original": original_b64,
            "mask": mask_b64,
            "result": result_b64,
            "message": "Object removal completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in object removal: {e}")
        return jsonify({"error": f"Object removal failed: {str(e)}"}), 500

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    info = {
        "model_loaded": model is not None,
        "device": str(device),
        "model_type": "Partial Convolution UNet" if model is not None else "None"
    }
    return jsonify(info)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
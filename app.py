from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import base64

app = Flask(__name__)

GALLERY_FOLDER = 'static/gallery'
os.makedirs(GALLERY_FOLDER, exist_ok=True)

# Image Generators
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

# Call this on start to ensure images exist
save_generated_images()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gallery')
def gallery():
    images = [f'/static/gallery/{img}' for img in os.listdir(GALLERY_FOLDER) if img.endswith('.png')]
    return render_template('gallery.html', images=images)

# Simple grayscale and sobel edge detection on uploaded image
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

if __name__ == "__main__":
    app.run(debug=True)

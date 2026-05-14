from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': file.filename})

@app.route('/filter', methods=['POST'])
def apply_filter():
    data = request.get_json()
    filename = data['filename']
    filter_type = data['filter']
    params = data.get('params', {})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(filepath)
    
    if filter_type == 'box':
        ksize = params.get('ksize', 3)
        filtered = cv2.boxFilter(img, -1, (ksize, ksize))
    elif filter_type == 'gaussian':
        ksize = params.get('ksize', 3)
        sigma = params.get('sigma', 0)
        filtered = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    elif filter_type == 'median':
        ksize = params.get('ksize', 3)
        filtered = cv2.medianBlur(img, ksize)
    elif filter_type == 'sobel':
        dx = params.get('dx', 1)
        dy = params.get('dy', 0)
        ksize = params.get('ksize', 3)
        filtered = cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)
        filtered = cv2.convertScaleAbs(filtered)
    else:
        return jsonify({'error': 'Invalid filter type'})
    
    output_filename = f"filtered_{filter_type}_{filename}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, filtered)
    
    return jsonify({'success': True, 'filtered_filename': output_filename})

@app.route('/gradient', methods=['POST'])
def calculate_gradient():
    data = request.get_json()
    filename = data['filename']
    x1 = data['x1']
    y1 = data['y1']
    x2 = data['x2']
    y2 = data['y2']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    # Extract region of interest
    roi = img[y1:y2, x1:x2]
    
    # Calculate gradients
    grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    
    # Calculate average direction
    avg_direction = np.mean(direction)
    
    return jsonify({
        'success': True,
        'avg_direction': float(avg_direction),
        'magnitude_mean': float(np.mean(magnitude))
    })

@app.route('/transform', methods=['POST'])
def transform_image():
    data = request.get_json()
    filename = data['filename']
    angle = data['angle']
    dx = data['dx']
    dy = data['dy']
    scale = data['scale']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(filepath)
    
    # Get image center
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create transformation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    
    # Apply transformation
    transformed = cv2.warpAffine(img, M, (w, h))
    
    # Save transformed image
    transformed_filename = f"transformed_{filename}"
    transformed_path = os.path.join(app.config['UPLOAD_FOLDER'], transformed_filename)
    cv2.imwrite(transformed_path, transformed)
    
    return jsonify({
        'success': True,
        'transformed_filename': transformed_filename
    })

@app.route('/frequency', methods=['POST'])
def frequency_filter():
    data = request.get_json()
    filename = data['filename']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    # Fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Save spectrum
    spectrum_filename = f"spectrum_{filename}"
    spectrum_path = os.path.join(app.config['UPLOAD_FOLDER'], spectrum_filename)
    cv2.imwrite(spectrum_path, magnitude_spectrum.astype(np.uint8))
    
    # Inverse Fourier transform
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Save reconstructed image
    recon_filename = f"recon_{filename}"
    recon_path = os.path.join(app.config['UPLOAD_FOLDER'], recon_filename)
    cv2.imwrite(recon_path, img_back.astype(np.uint8))
    
    return jsonify({
        'success': True,
        'spectrum_filename': spectrum_filename,
        'recon_filename': recon_filename
    })

if __name__ == '__main__':
    app.run(debug=True)
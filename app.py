from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
import uuid
import time
from preprocessing import extract_frames, extract_face, preprocess_image
from explainability import compute_gradcam, create_heatmap_overlay
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('best_model.h5')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}


# Import preprocessing functions
from preprocessing import extract_frames, extract_face, preprocess_image


def process_video(video_path, max_frames=20, target_size=(224, 224)):
    """Process video for deepfake detection."""
    # Create a temporary directory for frames
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)

    # Extract frames
    extract_frames(video_path, temp_dir)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Process frames
    frames = []
    frame_paths = []

    for frame_file in sorted(os.listdir(temp_dir))[:max_frames]:
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(temp_dir, frame_file)
            face = extract_face(frame_path, face_cascade, target_size)
            if face is not None:
                face = face / 255.0
                frames.append(face)
                frame_paths.append(frame_path)

    # Ensure we have at least one frame
    if len(frames) == 0:
        return None, None

    # Ensure we have exactly max_frames frames
    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))

    # Trim to max_frames
    frames = frames[:max_frames]
    frame_paths = frame_paths[:min(len(frame_paths), max_frames)]

    return np.array([frames]), frame_paths


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process file based on type
        if is_video(filename):
            X, frame_paths = process_video(filepath)
            if X is None:
                return jsonify({'error': 'No faces detected in video'})
        else:
            # Process image
            X = preprocess_image(filepath)
            frame_paths = [filepath]

        # Make prediction
        prediction = model.predict(X)
        score = float(prediction[0][0])

        # Compute GradCAM for the first frame
        layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.TimeDistributed):
                layer_name = layer.name
                break

        # Generate heatmap for the first frame
        from explainability import compute_gradcam, create_heatmap_overlay

        if is_video(filename):
            sample = X[0][0]  # First frame of the video
        else:
            sample = X[0]  # Single image

        cam = compute_gradcam(model, np.expand_dims(sample, axis=0), layer_name, class_idx=int(score > 0.5))

        # Save heatmap
        heatmap_overlay = create_heatmap_overlay(sample, cam)
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_' + filename.split('.')[0] + '.jpg')
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))

        processing_time = time.time() - start_time

        return jsonify({
            'score': score,
            'classification': 'Fake' if score > 0.5 else 'Real',
            'confidence': score if score > 0.5 else 1 - score,
            'processing_time': processing_time,
            'heatmap_url': '/uploads/heatmap_' + filename.split('.')[0] + '.jpg'
        })

    return jsonify({'error': 'File type not allowed'})


if __name__ == '__main__':
    app.run(debug=True)
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm


def extract_frames(video_path, output_dir, frames_per_second=1):
    """Extract frames from a video file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second)

    success, frame = video.read()
    count = 0
    frame_count = 0

    while success:
        if count % frame_interval == 0:
            # Save frame as JPEG file
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        success, frame = video.read()
        count += 1

    video.release()
    return frame_count


def extract_face(image_path, face_detector, target_size=(224, 224)):
    """Extract face from an image and resize to target size."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_detector.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # If no face detected, return resized original image
    if len(faces) == 0:
        return cv2.resize(image_rgb, target_size)

    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # Add margin to face
    margin = int(0.2 * w)
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(image.shape[1], x + w + margin)
    y_end = min(image.shape[0], y + h + margin)

    face = image_rgb[y_start:y_end, x_start:x_end]

    # Resize to target size
    face = cv2.resize(face, target_size)

    return face


def preprocess_data(data_dir, target_size=(224, 224), max_frames=20):
    """Preprocess data for the model."""
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    X = []
    y = []

    # Process real videos
    real_dir = os.path.join(data_dir, 'real')
    for video in tqdm(os.listdir(real_dir)):
        if video.endswith('.mp4') or video.endswith('.avi'):
            video_path = os.path.join(real_dir, video)
            frames_dir = os.path.join(data_dir, 'extracted_frames', 'real', video.split('.')[0])

            # Extract frames if not already done
            if not os.path.exists(frames_dir):
                extract_frames(video_path, frames_dir)

            # Process frames
            frames = []
            for frame_file in sorted(os.listdir(frames_dir))[:max_frames]:
                if frame_file.endswith('.jpg'):
                    frame_path = os.path.join(frames_dir, frame_file)
                    face = extract_face(frame_path, face_cascade, target_size)
                    if face is not None:
                        face = img_to_array(face) / 255.0
                        frames.append(face)

            # Ensure we have exactly max_frames frames
            if len(frames) > 0:
                while len(frames) < max_frames:
                    frames.append(np.zeros_like(frames[0]))

                X.append(frames[:max_frames])
                y.append(0)  # 0 for real

    # Process fake videos
    fake_dir = os.path.join(data_dir, 'fake')
    for video in tqdm(os.listdir(fake_dir)):
        if video.endswith('.mp4') or video.endswith('.avi'):
            video_path = os.path.join(fake_dir, video)
            frames_dir = os.path.join(data_dir, 'extracted_frames', 'fake', video.split('.')[0])

            # Extract frames if not already done
            if not os.path.exists(frames_dir):
                extract_frames(video_path, frames_dir)

            # Process frames
            frames = []
            for frame_file in sorted(os.listdir(frames_dir))[:max_frames]:
                if frame_file.endswith('.jpg'):
                    frame_path = os.path.join(frames_dir, frame_file)
                    face = extract_face(frame_path, face_cascade, target_size)
                    if face is not None:
                        face = img_to_array(face) / 255.0
                        frames.append(face)

            # Ensure we have exactly max_frames frames
            if len(frames) > 0:
                while len(frames) < max_frames:
                    frames.append(np.zeros_like(frames[0]))

                X.append(frames[:max_frames])
                y.append(1)  # 1 for fake

    return np.array(X), np.array(y)


# Function to load and preprocess image (for web interface)
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image for the model."""
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Extract face
    face = extract_face(image_path, face_cascade, target_size)
    if face is None:
        # If no face detected, load and resize the original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = cv2.resize(image, target_size)

    # Normalize
    face = img_to_array(face) / 255.0

    return np.expand_dims(face, axis=0)  # Add batch dimension
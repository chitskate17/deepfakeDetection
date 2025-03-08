import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Import modules
from preprocessing import preprocess_data
from models import build_model, build_mesonet
from training import train_model, plot_training_history
from evaluation import evaluate_model, visualize_predictions, analyze_failure_cases
from explainability import visualize_gradcam


def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection System')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'web'],
                        help='Mode: train, evaluate, or web')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing training data')
    parser.add_argument('--model_path', type=str, default='best_model.h5',
                        help='Path to save or load model')
    parser.add_argument('--model_type', type=str, default='efficientnet_lstm',
                        choices=['efficientnet_lstm', 'mesonet'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--frames', type=int, default=20,
                        help='Number of frames to use for video processing')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Loading and preprocessing data...")
        X, y = preprocess_data(args.data_dir, max_frames=args.frames)

        print(f"Data loaded: {X.shape}, {y.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Building model...")
        if args.model_type == 'efficientnet_lstm':
            model = build_model(input_shape=(args.frames, 224, 224, 3))
        else:
            model = build_mesonet(input_shape=(224, 224, 3))

        print("Training model...")
        history, model = train_model(X_train, y_train, model, epochs=args.epochs, batch_size=args.batch_size)

        # Plot training history
        plot_training_history(history)

        # Save model
        model.save(args.model_path)
        print(f"Model saved to {args.model_path}")

        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print("Evaluation metrics:", metrics)

        # Visualize predictions
        print("Visualizing predictions...")
        visualize_predictions(model, X_test, y_test, None, num_samples=5)

        # Analyze failure cases
        print("Analyzing failure cases...")
        analyze_failure_cases(model, X_test, y_test)

        # Visualize GradCAM
        print("Visualizing GradCAM...")
        visualize_gradcam(model, X_test[:5], y_test[:5])

    elif args.mode == 'evaluate':
        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path)

        print("Loading test data...")
        X, y = preprocess_data(args.data_dir, max_frames=args.frames)

        # Split data
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print("Evaluation metrics:", metrics)

        # Visualize predictions
        print("Visualizing predictions...")
        visualize_predictions(model, X_test, y_test, None, num_samples=5)

        # Visualize GradCAM
        print("Visualizing GradCAM...")
        visualize_gradcam(model, X_test[:5], y_test[:5])

    elif args.mode == 'web':
        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path)

        print("Starting web interface...")
        from app import app
        app.run(debug=True)


if __name__ == '__main__':
    main()
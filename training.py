from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(X, y, model, epochs=20, batch_size=16, validation_split=0.2):
    """
    Train the model with appropriate callbacks.

    Args:
        X: Training features
        y: Training labels
        model: Compiled model
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Proportion of data to use for validation

    Returns:
        History object and trained model
    """
    # Set seed for reproducibility
    set_seed()

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )

    # Create callbacks
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Train the model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")

    # Load the best model
    model.load_weights('best_model.h5')

    return history, model


def plot_training_history(history):
    """Plot training & validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Plot AUC, Precision, Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['auc'], label='Train AUC')
    ax.plot(history.history['val_auc'], label='Val AUC')
    ax.plot(history.history['precision'], label='Train Precision')
    ax.plot(history.history['val_precision'], label='Val Precision')
    ax.plot(history.history['recall'], label='Train Recall')
    ax.plot(history.history['val_recall'], label='Val Recall')

    ax.set_title('Model Metrics')
    ax.set_ylabel('Score')
    ax.set_xlabel('Epoch')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
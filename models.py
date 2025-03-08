import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.models import Model


def build_model(input_shape=(20, 224, 224, 3), dropout_rate=0.5):
    """
    Build an EfficientNet+LSTM model for deepfake detection.

    Args:
        input_shape: Shape of input tensor (frames, height, width, channels)
        dropout_rate: Dropout rate for regularization

    Returns:
        Model: Compiled Keras model
    """
    # Input layer
    sequence_input = tf.keras.Input(shape=input_shape)

    # Load pre-trained EfficientNet with weights
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Apply the base model to each frame in the sequence
    encoded_frames = TimeDistributed(base_model)(sequence_input)

    # Add a bidirectional LSTM to capture temporal features
    x = Bidirectional(LSTM(256, return_sequences=True))(encoded_frames)
    x = Bidirectional(LSTM(128))(x)

    # Add dropout for regularization
    x = Dropout(dropout_rate)(x)

    # Add dense layers for classification
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    # Create the model
    model = Model(sequence_input, outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


# Alternative: MesoNet implementation
def build_mesonet(input_shape=(224, 224, 3)):
    """
    Build a MesoNet model for deepfake detection.

    Args:
        input_shape: Shape of input image (height, width, channels)

    Returns:
        Model: Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape)

    # First conv block
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Second conv block
    x = tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Third conv block
    x = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Fourth conv block
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
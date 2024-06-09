import tensorflow as tf
from tensorflow import keras
#from src.config.params import TARGET_SIZE

def build_model(input_shape: tuple[int, int, int] = (3840, 2160, 3), size_boxes: int = 4, num_objects: int = 8) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input")

    x = keras.layers.Conv2D(64, (10, 10), activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(64, (10, 10), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(128, (7, 7), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(128, (7, 7), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    # Flatten the output before Dense layers
    x = keras.layers.Flatten()(x)

    # Dense Layers
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    bbox_branch = keras.layers.Dense(256, activation='relu')(x)
    bbox_branch = keras.layers.Dropout(0.5)(bbox_branch)
    bbox_output = keras.layers.Dense(size_boxes * num_objects, activation='sigmoid', name='bbox_output')(bbox_branch)
    bbox_output = keras.layers.Reshape((num_objects, size_boxes), name='bbox_reshape')(bbox_output)

    model = keras.models.Model(inputs=inputs, outputs=[bbox_output])

    return model

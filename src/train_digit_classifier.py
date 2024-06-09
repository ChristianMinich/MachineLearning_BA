import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle

from src.config.params import CSV_FILE, IMAGE_DIRECTORY, FILENAME, EPOCHS
from src.utils.load_data import load_image_and_annotations, adjust_dataset_structure
from src.utils.load_data_digits import load_digit_and_annotations
from src.utils.dataset_tools import split_dataset
from src.plotting.plot_barcodes import plot_single_barcode_bounding_boxes, \
    plot_images_with_bboxes_from_dataset_with_range


def main():
    dataset: tf.data.Dataset = load_digit_and_annotations()
    for element in dataset.take(1):
        print(element)

    # Split Dataset
    dataset_size = len(list(dataset))
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset = dataset.take(train_size).batch(32)
    val_dataset = dataset.skip(train_size).take(val_size).batch(32)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
                                           monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks)

    save_history(history)
    save_model(model)

    loss, accuracy = model.evaluate(val_dataset)
    print(f'Validation accuracy: {accuracy:.4f}')

    plot_evaluations(history)


def plot_evaluations(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    # Plotting training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    #plt.show()

def save_history(history: tf.keras.callbacks.History) -> None:
    if not os.path.exists('histories'):
        os.makedirs('histories')

    with open(f'histories/{FILENAME}.pkl', 'wb') as f:
        pickle.dump(history.history, f)


def save_model(model: tf.keras.Model) -> None:
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save(os.path.join('models', f'{FILENAME}.keras'), save_format="keras")


if __name__ == '__main__':
    main()

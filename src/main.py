import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pickle

from config.params import CSV_FILE, IMAGE_DIRECTORY, FILENAME, EPOCHS, TARGET_SIZE
from train_digit_classifier import save_history
from utils.load_data import load_image_and_annotations, adjust_dataset_structure
from utils.dataset_tools import split_dataset
from model.build_model import build_model
from src.plotting.plot_barcodes import plot_single_barcode_bounding_boxes, \
    plot_images_with_bboxes_from_dataset_with_range


def main():
    annotations: pd.DataFrame = pd.read_csv(CSV_FILE)
    print(annotations.columns)

    dataset: tf.data.Dataset = load_image_and_annotations(
        annotations=annotations,
        skip_barcodes=False
    )
    print(dataset.take(1))

    def filter_strichcode(image, labels, bboxes):
        strichcode_class = tf.constant('Strichcode', dtype=tf.string)
        class_labels = tf.strings.as_string(labels[:, 0])
        mask = class_labels != strichcode_class
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)

        num_objects = 8
        padding_size_labels = num_objects - tf.shape(labels)[0]
        padding_size_bboxes = num_objects - tf.shape(bboxes)[0]

        padding_size_labels = tf.maximum(padding_size_labels, 0)
        padding_size_bboxes = tf.maximum(padding_size_bboxes, 0)

        labels = tf.pad(labels, [[0, padding_size_labels], [0, 0]])
        bboxes = tf.pad(bboxes, [[0, padding_size_bboxes], [0, 0]])

        return image, labels, bboxes

    filtered_dataset = dataset.map(lambda image, labels, bboxes: filter_strichcode(image, labels, bboxes))
    filtered_dataset = filtered_dataset.filter(
        lambda image, labels, bboxes: tf.shape(labels)[0] == 8)

    dataset = adjust_dataset_structure(filtered_dataset)
    dataset = dataset.cache()

    dataset = dataset.shuffle(buffer_size=len(annotations['filename'].unique()), reshuffle_each_iteration=False)

    for element in dataset.take(1):
        print('shuffled: ', element)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    def print_dataset(dataset, name):
        print(f"{name} Dataset:")
        for element in dataset.take(1):
            print(element)
            try:
                image, labels, bboxes = element
                print("Image shape:", image.shape)
                print("Labels shape:", labels.shape)
                print("Bboxes shape:", bboxes.shape)
            except ValueError as e:
                print(f"Error unpacking element: {e}")

    print_dataset(train_dataset, "Train")
    print_dataset(val_dataset, "Validation")
    print_dataset(test_dataset, "Test")

    model: tf.keras.Model = build_model(size_boxes=4, num_objects=8)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
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


def save_history(history: tf.keras.callbacks.History) -> None:
    if not os.path.exists('histories'):
        os.makedirs('histories')

    with open(f'histories/{FILENAME}.pkl', 'wb') as f:
        pickle.dump(history.history, f)


def save_model(model: tf.keras.Model) -> None:
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save(os.path.join('models', f'{FILENAME}.keras'), save_format="keras")


def plot_evaluations(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    # plt.show()


if __name__ == '__main__':
    main()

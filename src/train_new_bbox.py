import tensorflow as tf
import pandas as pd#
import os
import matplotlib.pyplot as plt
import pickle
import keras

from utils.load_data_thomas import load_and_augment_images_for_digit_bbox
from config.params import EPOCHS, FILENAME, TARGET_SIZE4DIGIT_BBOX

FILENAME = FILENAME + f'_TS{TARGET_SIZE4DIGIT_BBOX}_digit_bbox_delta07_lr1e-4_more_Dense'

def build_model(input_shape=(TARGET_SIZE4DIGIT_BBOX, TARGET_SIZE4DIGIT_BBOX, 3), size_boxes=4, num_objects=7):
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

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    bbox_branch = keras.layers.Dense(256, activation='relu')(x)
    bbox_branch = keras.layers.Dropout(0.5)(bbox_branch)
    bbox_output = keras.layers.Dense(size_boxes * num_objects, activation='sigmoid', name='bbox_output')(bbox_branch)
    bbox_output = keras.layers.Reshape((num_objects, size_boxes), name='bbox_reshape')(bbox_output)

    model = keras.models.Model(inputs=inputs, outputs=[bbox_output])

    return model

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

def save_history(history):
    if not os.path.exists('histories'):
        os.makedirs('histories')

    with open(f'histories/{FILENAME}.pkl', 'wb') as f:
        pickle.dump(history.history, f)

def save_model(model):
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save(os.path.join('models', f'{FILENAME}.keras'), save_format="keras")

def train_bbox():
    csv_file_path = '/home/chminich/MachineLearning_BA/data/Barcodes_v10i/_annotations.csv'
    image_base_path = '/home/chminich/MachineLearning_BA/data/Barcodes_v10i/'

    annotations = pd.read_csv(csv_file_path)

    df_filtered = annotations[annotations['class'] != 'Strichcode']

    dataset = load_and_augment_images_for_digit_bbox(df_filtered, image_base_path, skip_barcodes=True)

    train_dataset, val_dataset = keras.utils.split_dataset(dataset, left_size=0.8, right_size=0.2, shuffle=True)

    '''
    # Filter out None values explicitly
    def filter_none(data):
        for key, value in data.items():
            if value is None:
                print(f"Found None value in key: {key}")
                return False
        return True

    dataset = dataset.filter(filter_none)

    # Check for None values in the dataset
    for data in dataset:
        if any(value is None for value in data.values()):
            print("None value detected in dataset batch")
        else:
            print("No None values in dataset batch")

    # Determine dataset size more efficiently
    dataset_size = dataset.cardinality().numpy()
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    print(f"Train dataset size: {train_size}")
    print(f"Validation dataset size: {val_size}")

    # Verify the first batch of train and validation datasets
    for data in train_dataset.take(1):
        print("Train dataset batch example:", data)

    for data in val_dataset.take(1):
        print("Validation dataset batch example:", data)

    '''
    # Final check for None values before training
    def final_check_for_none(dataset, dataset_name):
        for batch in dataset:
            for key, value in batch.items():
                if value is None:
                    print(f"Found None value in {dataset_name} dataset in key: {key}")
                    return False
        return True

    if not final_check_for_none(train_dataset, "train"):
        print("Training aborted due to None values in train dataset.")
        return

    if not final_check_for_none(val_dataset, "validation"):
        print("Training aborted due to None values in validation dataset.")
        return

    model = build_model(input_shape=(TARGET_SIZE4DIGIT_BBOX, TARGET_SIZE4DIGIT_BBOX, 3), size_boxes=4, num_objects=7)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'best_model_bbox.keras'),
                                           monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks)

    save_history(history)
    save_model(model)

    loss, accuracy = model.evaluate(val_dataset)
    print(f'Validation accuracy: {accuracy:.4f}')

    plot_evaluations(history)

if __name__ == '__main__':
    train_bbox()
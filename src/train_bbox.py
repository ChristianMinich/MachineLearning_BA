import tensorflow as tf
import pandas as pd
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle

EPOCHS = 10
FILENAME = 'bbox_detector_model_' + str(EPOCHS)

class_label_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'Aufkleber': 10
}

def consolidate_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    images = df['filename'].unique()

    consolidated_data = []
    for img in images:
        img_df = df[df['filename'] == img]

        classes = []
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []

        for _, row in img_df.iterrows():
            class_label = row['class']
            if class_label == 'Strichcode':
                continue  # Skip 'Strichcode'
            classes.append(class_label_mapping.get(class_label, -1))
            xmins.append(row['xmin'])
            ymins.append(row['ymin'])
            xmaxs.append(row['xmax'])
            ymaxs.append(row['ymax'])

        # Ensure each list has exactly 8 bounding boxes by padding with 0 if necessary
        while len(classes) < 8:
            classes.append(0)
            xmins.append(0)
            ymins.append(0)
            xmaxs.append(0)
            ymaxs.append(0)

        consolidated_data.append({
            'filename': img,
            'classes': classes[:8],
            'xmins': xmins[:8],
            'ymins': ymins[:8],
            'xmaxs': xmaxs[:8],
            'ymaxs': ymaxs[:8]
        })

        print(f"Processed image: {img}")

    return consolidated_data

def resize_and_pad_image(image, target_size=(3840, 2160)):
    original_shape = tf.shape(image)[:2]
    height, width = tf.cast(original_shape[0], tf.float32), tf.cast(original_shape[1], tf.float32)

    scale = tf.reduce_min(target_size / tf.cast([height, width], tf.float32))
    new_height, new_width = tf.cast(height * scale, tf.int32), tf.cast(width * scale, tf.int32)

    image = tf.image.resize(image, (new_height, new_width))

    delta_height, delta_width = target_size[0] - new_height, target_size[1] - new_width
    top, bottom = tf.cast(delta_height // 2, tf.int32), tf.cast(delta_height - (delta_height // 2), tf.int32)
    left, right = tf.cast(delta_width // 2, tf.int32), tf.cast(delta_width - (delta_width // 2), tf.int32)

    image = tf.image.pad_to_bounding_box(image, top, left, target_size[0], target_size[1])
    return image

def parse_data(data_row, image_base_path, target_size=(3840, 2160)):
    image_path = os.path.join(image_base_path, data_row['filename'])

    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None

    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = resize_and_pad_image(image, target_size)
        image = tf.cast(image, tf.float32) / 255.0
        print(f"Successfully loaded and processed image: {image_path}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    return {
        'input': image,
        'width': tf.constant(target_size[1], dtype=tf.int32),
        'height': tf.constant(target_size[0], dtype=tf.int32),
        'class': tf.constant(data_row['classes'], dtype=tf.int64),
        'xmin': tf.constant(data_row['xmins'], dtype=tf.float32),
        'ymin': tf.constant(data_row['ymins'], dtype=tf.float32),
        'xmax': tf.constant(data_row['xmaxs'], dtype=tf.float32),
        'ymax': tf.constant(data_row['ymaxs'], dtype=tf.float32),
    }

def load_data_to_tf_dataset(csv_file_path, image_base_path, target_size=(3840, 2160), batch_size=2):
    consolidated_data = consolidate_data(csv_file_path)

    def generator():
        for data_row in consolidated_data:
            parsed_data = parse_data(data_row, image_base_path, target_size)
            if parsed_data is not None:
                yield parsed_data
            else:
                print(f"Skipping image due to load failure: {data_row['filename']}")

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            'input': tf.TensorSpec(shape=target_size + (3,), dtype=tf.float32),
            'width': tf.TensorSpec(shape=(), dtype=tf.int32),
            'height': tf.TensorSpec(shape=(), dtype=tf.int32),
            'class': tf.TensorSpec(shape=(8,), dtype=tf.int64),
            'xmin': tf.TensorSpec(shape=(8,), dtype=tf.float32),
            'ymin': tf.TensorSpec(shape=(8,), dtype=tf.float32),
            'xmax': tf.TensorSpec(shape=(8,), dtype=tf.float32),
            'ymax': tf.TensorSpec(shape=(8,), dtype=tf.float32),
        }
    )
    dataset = dataset.batch(batch_size)
    return dataset

def build_model(input_shape=(3840, 2160, 3), size_boxes=4, num_objects=8):
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

    dataset = load_data_to_tf_dataset(csv_file_path, image_base_path, batch_size=4)  # Increased batch size to 4

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=100)

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

    model = build_model(input_shape=(3840, 2160, 3), size_boxes=4, num_objects=8)

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

    # Save evaluations in a plot
    plot_evaluations(history)

if __name__ == '__main__':
    train_bbox()

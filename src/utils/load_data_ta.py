from random import randint

from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tensorflow.keras as keras
from keras.src.utils import to_categorical
import matplotlib.pyplot as plt

from config.params_ta import TARGET_SIZE, CLASS_TO_INDEX, BATCH_SIZE, IMAGE_DIRECTORY, TARGET_SIZE4DIGIT, TARGET_SIZE4DIGIT_BBOX


def load_image_and_annotations(annotations: pd.DataFrame, image_dir: str = IMAGE_DIRECTORY,
                               num_objects: int = 9, size_boxes: int = 4,
                               num_classes: int = len(CLASS_TO_INDEX),
                               skip_barcodes: bool = False, skip_trailer: bool = False) -> tf.data.Dataset:
    """
    Returns a dataset of images, bounding boxes and classes
    :param annotations:
    :param image_dir:
    :param num_objects:
    :param size_boxes:
    :param num_classes:
    :param skip_barcodes:
    :param skip_trailer:
    :return:
    """
    image_paths = []
    all_boxes = []
    all_classes = []

    grouped = annotations.groupby('filename')

    for filename, group in grouped:
        image_path = os.path.join(image_dir, filename)

        if skip_trailer and str(filename).startswith('IMG'):
            print('skipping:', filename)
            continue

        if skip_barcodes and str(filename).startswith('barcode'):
            print('skipping:', filename)
            continue

        if not os.path.exists(image_path):
            continue

        # Initialize inputs with zeros
        boxes = np.zeros((num_objects, size_boxes))
        classes = np.zeros((num_objects, num_classes))

        for i, (_, row) in enumerate(group.iterrows()):
            if i >= num_objects:
                break
            x_min = row['xmin']
            y_min = row['ymin']
            x_max = row['xmax']
            y_max = row['ymax']
            boxes[i] = [y_min, x_min, y_max, x_max]
            classes[i] = tf.one_hot(CLASS_TO_INDEX[str(row['class'])], depth=num_classes)

        image_paths.append(image_path)
        all_boxes.append(boxes)
        all_classes.append(classes)

    boxes_tensor = tf.convert_to_tensor(all_boxes, dtype=tf.float32)
    classes_tensor = tf.convert_to_tensor(all_classes, dtype=tf.float32)

    for i, box in enumerate(boxes_tensor):
        assert box.shape == (num_objects, size_boxes), \
            f"Expected shape ({num_objects}, {size_boxes}) for sample {i}, but got {box.shape}"

    for i, classes in enumerate(classes_tensor):
        assert classes.shape == (num_objects, num_classes), \
            f"Expected shape ({num_objects}, {num_classes}) for sample {i}, but got {classes.shape}"

    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    boxes_ds = tf.data.Dataset.from_tensor_slices(boxes_tensor)
    classes_ds = tf.data.Dataset.from_tensor_slices(classes_tensor)

    combined_dataset = tf.data.Dataset.zip((image_paths_ds, boxes_ds, classes_ds))

    images_dataset = combined_dataset.flat_map(
        lambda image_path, boxes, classes: process_element(image_path, boxes, classes))

    return images_dataset


def load_and_crop_image(filename, xmin, ymin, xmax, ymax):
    image = Image.open(filename)
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    """image = np.array(cropped_image)
    plt.imshow(image)
    plt.show()"""
    return cropped_image


def preprocess_image(original_image: Image, target_size: int):
    image = np.array(original_image)
    image = tf.image.resize_with_pad(image, target_size, target_size)

    original_shape: tf.Tensor = tf.shape(original_image)

    image = image / 255.0
    assert image.shape == (target_size, target_size, 3)

    return image, original_shape


def load_images_for_digit_bbox(annotations: pd.DataFrame, image_dir: str = IMAGE_DIRECTORY,
                               skip_barcodes: bool = False, skip_trailer: bool = False) -> tf.data.Dataset:
    grouped = annotations.groupby('filename')

    images = []
    bboxes = []
    i = 0

    for filename, group in grouped:
        i += 1
        image_path = os.path.join(image_dir, filename)

        if skip_trailer and str(filename).startswith('IMG'):
            print('skipping:', filename)
            continue

        if skip_barcodes and str(filename).startswith('bar'):
            print('skipping:', filename)
            continue

        if not os.path.exists(image_path):
            continue

        aufkleber_row = group[group['class'] == 'Aufkleber']

        if aufkleber_row.empty:
            continue

        aufkleber_row = aufkleber_row.iloc[0]
        aufkleber_xmin, aufkleber_ymin = (aufkleber_row['xmin'] - randint(0, 50),
                                          aufkleber_row['ymin'] - randint(0, 50))
        aufkleber_xmax, aufkleber_ymax = (aufkleber_row['xmax'] + randint(0, 50),
                                          aufkleber_row['ymax'] + randint(0, 50))

        cropped_image = load_and_crop_image(image_path, aufkleber_xmin, aufkleber_ymin,
                                            aufkleber_xmax, aufkleber_ymax)

        short_bbox = []

        for _, row in group.iterrows():
            if row['class'] == 'Aufkleber':
                continue

            xmin = max(0, row['xmin'] - aufkleber_xmin)
            ymin = max(0, row['ymin'] - aufkleber_ymin)
            xmax = max(0, row['xmax'] - aufkleber_xmin)
            ymax = max(0, row['ymax'] - aufkleber_ymin)

            short_bbox.append([ymin, xmin, ymax, xmax])

        preprocessed_image, original_shape = preprocess_image(cropped_image, TARGET_SIZE4DIGIT_BBOX)

        adjusted_bboxes = adjust_boxes_for_padding(tf.convert_to_tensor(short_bbox, dtype=tf.float32),
                                                   tf.cast(original_shape[0], tf.float32),
                                                   tf.cast(original_shape[1], tf.float32),
                                                   target_size=TARGET_SIZE4DIGIT_BBOX)

        images.append(preprocessed_image)
        bboxes.append(adjusted_bboxes)

        # if i > 100:
        #    break  # comment for visualization
    if not images:
        raise ValueError("No images or labels found.")

    print(f'Loaded {len(images)} images')

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images, bboxes))

    return dataset


def load_images_for_digit_detection(annotations: pd.DataFrame, image_dir: str = IMAGE_DIRECTORY, num_classes: int = 10,
                                    skip_barcodes: bool = False, skip_trailer: bool = False) -> tf.data.Dataset:
    grouped = annotations.groupby('filename')

    images = []
    labels = []

    for filename, group in grouped:

        image_path = os.path.join(image_dir, filename)

        if skip_trailer and str(filename).startswith('IMG'):
            print('skipping:', filename)
            continue

        if skip_barcodes and str(filename).startswith('bar'):
            print('skipping:', filename)
            continue

        if not os.path.exists(image_path):
            continue

        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            label = row['class']
            image = load_and_crop_image(image_path, xmin, ymin, xmax, ymax)
            image, __ = preprocess_image(image, TARGET_SIZE4DIGIT)
            one_hot = to_categorical(label, num_classes=num_classes)
            images.append(image)
            labels.append(one_hot)
        # break  # comment for visualization
    if not images or not labels:
        raise ValueError("No images or labels found.")

    print(f'Loaded {len(images)} images and {len(labels)} labels')

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    return dataset


def remove_classes(dataset: tf.data.Dataset) -> tf.data.Dataset:
    def map_remove_classes(image, boxes, classes):
        return image, boxes

    return dataset.map(map_remove_classes)


def load_digits_and_annotations(annotations: pd.DataFrame, image_dir: str,
                                num_objects: int = 7, num_classes: int = len(CLASS_TO_INDEX),
                                skip_barcodes: bool = False, skip_trailer: bool = False) -> tf.data.Dataset:
    image_paths = []
    all_classes = []

    grouped = annotations.groupby('filename')

    for filename, group in grouped:
        image_path = os.path.join(image_dir, filename)

        if skip_trailer and filename.startswith('IMG'):
            print('skipping:', filename)
            continue

        if skip_barcodes and filename.startswith('barcode'):
            print('skipping:', filename)
            continue

        if not os.path.exists(image_path):
            continue

        classes = np.zeros((num_objects, num_classes))

        for i, (_, row) in enumerate(group.iterrows()):
            if i < num_objects:
                classes[i] = tf.one_hot(CLASS_TO_INDEX[str(row['class'])], depth=num_classes).numpy()

        image_paths.append(image_path)
        all_classes.append(classes)

    classes_tensor = tf.convert_to_tensor(all_classes, dtype=tf.float32)

    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    classes_ds = tf.data.Dataset.from_tensor_slices(classes_tensor)

    combined_dataset = tf.data.Dataset.zip((image_paths_ds, classes_ds))

    images_dataset = combined_dataset.flat_map(
        lambda image_path, classes: process_element_without_boxes(image_path, classes)
    )

    return images_dataset


def process_element_without_boxes(image_path: str, classes: tf.Tensor) -> tf.data.Dataset:
    original_image, augmented_brightness_image, augmented_contrast_image = \
        load_process_and_augment_without_boxes(image_path)
    return flatten_dataset_without_boxes(original_image, augmented_brightness_image,
                                         augmented_contrast_image, classes)


def flatten_dataset_without_boxes(original_image: tf.Tensor, augmented_brightness_image: tf.Tensor,
                                  augmented_contrast_image: tf.Tensor,
                                  classes: tf.Tensor) -> tf.data.Dataset:
    num_objects = classes.shape[0]
    classes_list = [classes[i] for i in range(num_objects)]
    original_ds = tf.data.Dataset.from_tensors((original_image, tuple(classes_list)))
    brightness_ds = tf.data.Dataset.from_tensors((augmented_brightness_image, tuple(classes_list)))
    contrast_ds = tf.data.Dataset.from_tensors((augmented_contrast_image, tuple(classes_list)))
    return original_ds.concatenate(brightness_ds).concatenate(contrast_ds)


def load_process_and_augment_without_boxes(image_path: str) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    original_image = load_and_process_image_without_boxes(image_path)
    augmented_brightness_image = random_brightness(original_image)
    augmented_contrast_image = random_contrast(original_image)
    return original_image, augmented_brightness_image, augmented_contrast_image


def load_and_process_image_without_boxes(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, TARGET_SIZE, TARGET_SIZE)
    return image


def adjust_boxes_for_padding(boxes: tf.Tensor, original_height, original_width, target_size: int) -> tf.Tensor:
    # original_height = tf.cast(original_shape[0], tf.float32)
    # original_width = tf.cast(original_shape[1], tf.float32)

    scale_factor = target_size / tf.maximum(original_height, original_width)

    new_height = original_height * scale_factor
    new_width = original_width * scale_factor

    pad_y = (target_size - new_height) / 2.0
    pad_x = (target_size - new_width) / 2.0

    boxes = tf.stack([
        (boxes[:, 0] * scale_factor) + pad_y,
        (boxes[:, 1] * scale_factor) + pad_x,
        (boxes[:, 2] * scale_factor) + pad_y,
        (boxes[:, 3] * scale_factor) + pad_x,
    ], axis=-1)

    """for box in boxes:
        y_min, x_min, y_max, x_max = box
        if x_max - x_min > y_max - y_min:
            raise ValueError('shit happened')"""  # for debugging

    return boxes / target_size


def load_and_process_image(image_path: str, target_size: int = TARGET_SIZE) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    original_shape = tf.shape(image)

    image = tf.image.resize_with_pad(image, target_size, target_size)

    image = image / 255.0
    assert image.shape == (target_size, target_size, 3)

    return image, original_shape


def load_and_augment_images_for_digit_bbox(annotations: pd.DataFrame, image_dir: str = IMAGE_DIRECTORY,
                                           skip_barcodes: bool = False, skip_trailer: bool = False) -> tf.data.Dataset:
    dataset = load_images_for_digit_bbox(annotations, image_dir, skip_barcodes, skip_trailer)

    # Apply data augmentations separately
    brightness_augmented_dataset = apply_augmentation(dataset, random_brightness)
    contrast_augmented_dataset = apply_augmentation(dataset, random_contrast)

    # Concatenate original and augmented datasets
    combined_dataset = dataset.concatenate(brightness_augmented_dataset)
    combined_dataset = combined_dataset.concatenate(contrast_augmented_dataset)

    return combined_dataset


def load_process_and_augment(image_path: str, boxes: tf.Tensor, classes: tf.Tensor) \
        -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    original_image, original_shape = load_and_process_image(image_path)
    adjusted_boxes = adjust_boxes_for_padding(boxes, tf.cast(original_shape[0], tf.float32),
                                              tf.cast(original_shape[1], tf.float32), TARGET_SIZE)
    augmented_brightness_image = random_brightness(original_image)
    augmented_contrast_image = random_contrast(original_image)
    return original_image, augmented_brightness_image, augmented_contrast_image, adjusted_boxes, classes


def flatten_dataset(original_image: tf.Tensor, augmented_brightness_image: tf.Tensor,
                    augmented_contrast_image: tf.Tensor, boxes: tf.Tensor,
                    classes: tf.Tensor) -> tf.data.Dataset:
    original_ds = tf.data.Dataset.from_tensors((original_image, boxes, classes))
    brightness_ds = tf.data.Dataset.from_tensors((augmented_brightness_image, boxes, classes))
    contrast_ds = tf.data.Dataset.from_tensors((augmented_contrast_image, boxes, classes))
    return original_ds.concatenate(brightness_ds).concatenate(contrast_ds)


def process_element(image_path: str, boxes: tf.Tensor, classes: tf.Tensor) -> tf.data.Dataset:
    original_image, augmented_brightness_image, augmented_contrast_image, boxes, classes = \
        load_process_and_augment(image_path, boxes, classes)
    return flatten_dataset(original_image, augmented_brightness_image,
                           augmented_contrast_image, boxes, classes)


def random_brightness(image: tf.Tensor, max_delta: float = 0.3) -> tf.Tensor:
    return tf.image.random_brightness(image, max_delta)


def random_contrast(image: tf.Tensor, lower: float = 0.7, upper: float = 1.3) -> tf.Tensor:
    return tf.image.random_contrast(image, lower, upper)


def apply_augmentation(dataset: tf.data.Dataset, augmentation_fn) -> tf.data.Dataset:
    return dataset.map(lambda image, bbox: (augmentation_fn(image), bbox))


def augment(image: tf.Tensor) -> tf.Tensor:
    image = random_brightness(image)
    image = random_contrast(image)
    return image


def adjust_dataset_structure(dataset: tf.data.Dataset) -> tf.data.Dataset:
    def map_function(image, bbox_coordinates, class_labels):
        return image, (class_labels, bbox_coordinates)

    return dataset.map(map_function)


def split_dataset(dataset: tf.data.Dataset) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    data = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    data_list = list(data)
    total_size = len(data_list)
    train_size = int(total_size * .7)
    val_size = int(total_size * .2)
    test_size = total_size - train_size - val_size

    train: tf.data.Dataset = data.take(train_size)
    val: tf.data.Dataset = data.skip(train_size).take(val_size)
    test: tf.data.Dataset = data.skip(train_size + val_size).take(test_size)

    # Iterate over the train dataset to print the shapes
    for images, classes in train.take(5):
        print(images.shape)
        print(classes.shape)

    return train, val, test


def rotate_bb90deg(bbox, img_width):
    x_min, y_min, x_max, y_max = bbox
    new_xmin = y_min
    new_ymin = img_width - x_max
    new_xmax = y_max
    new_ymax = img_width - x_min
    return [new_xmin, new_ymin, new_xmax, new_ymax]

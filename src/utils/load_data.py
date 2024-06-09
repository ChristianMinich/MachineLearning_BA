from typing import Tuple, Union

import tensorflow as tf
import pandas as pd
import numpy as np
import os

from src.config.params import IMAGE_DIRECTORY, CLASS_TO_INDEX, TARGET_SIZE, BATCH_SIZE


def load_image_and_annotations(annotations: pd.DataFrame,
                               num_objects: int = 9, size_boxes: int = 4,
                               num_classes: int = len(CLASS_TO_INDEX),
                               skip_barcodes: bool = False, skip_trailer: bool = False) -> tf.data.Dataset:
    image_paths = []
    all_boxes = []
    all_classes = []

    grouped = annotations.groupby('filename')

    for filename, group in grouped:
        image_path = os.path.join(IMAGE_DIRECTORY, filename)

        if skip_trailer and filename.startswith('IMG'):
            print('skipping:', filename)
            continue

        if skip_barcodes and filename.startswith('barcode'):
            print('skipping:', filename)
            continue

        if not os.path.exists(image_path):
            continue

        boxes = np.zeros((num_objects, size_boxes))
        classes = np.zeros((num_objects, num_classes))

        for i, (_, row) in enumerate(group.iterrows()):
            if i >= num_objects:
                break  # Ensure we don't exceed the pre-allocated size

            x_min = row['xmin']
            y_min = row['ymin']
            x_max = row['xmax']
            y_max = row['ymax']
            boxes[i] = [y_min, x_min, y_max, x_max]
            classes[i] = tf.one_hot(CLASS_TO_INDEX[str(row['class'])], depth=num_classes).numpy()

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


def adjust_boxes_for_padding(boxes: tf.Tensor, original_shape: tf.Tensor, target_size: int) -> tf.Tensor:
    original_height = tf.cast(original_shape[0], tf.float32)
    original_width = tf.cast(original_shape[1], tf.float32)

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

    return boxes / target_size

def adjust_boxes_for_padding_with_tensor(boxes: tf.Tensor, original_shape: tf.Tensor, target_size: tf.Tensor) -> tf.Tensor:
    original_height = tf.cast(original_shape[0], tf.float32)
    original_width = tf.cast(original_shape[1], tf.float32)

    scale_factor = tf.minimum(target_size[0] / original_height, target_size[1] / original_width)

    new_height = original_height * scale_factor
    new_width = original_width * scale_factor

    pad_y = (target_size[0] - new_height) / 2.0
    pad_x = (target_size[1] - new_width) / 2.0

    boxes = tf.stack([
        (boxes[:, 0] * scale_factor) + pad_y,
        (boxes[:, 1] * scale_factor) + pad_x,
        (boxes[:, 2] * scale_factor) + pad_y,
        (boxes[:, 3] * scale_factor) + pad_x,
    ], axis=-1)

    return boxes / tf.stack([target_size[0], target_size[1], target_size[0], target_size[1]], axis=0)


def load_and_process_image(image_path: str, target_size: int = TARGET_SIZE) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    original_shape = tf.shape(image)

    image = tf.image.resize_with_pad(image, target_size, target_size)

    image = image / 255.0
    assert image.shape == (target_size, target_size, 3)

    return image, original_shape


def load_process_and_augment(image_path: str, boxes: tf.Tensor, classes: tf.Tensor) \
        -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    original_image, original_shape = load_and_process_image(image_path)
    adjusted_boxes = adjust_boxes_for_padding_with_tensor(boxes, original_shape, tf.constant([3840, 2160], dtype=tf.float32))
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

    total_size = sum(1 for _ in dataset)
    train_size = int(total_size * .7)
    val_size = int(total_size * .2)
    test_size = total_size - train_size - val_size

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    return train, val, test


def rotatebb90Deg(bndbox, img_width):
    x_min, y_min, x_max, y_max = bndbox
    new_xmin = y_min
    new_ymin = img_width - x_max
    new_xmax = y_max
    new_ymax = img_width - x_min
    return [new_xmin, new_ymin, new_xmax, new_ymax]

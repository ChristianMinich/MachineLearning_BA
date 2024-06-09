from typing import Tuple, List, Callable
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from src.config.params import IMAGE_DIRECTORY, CLASS_TO_INDEX, TARGET_SIZE

class DatasetBuilder:
    def __init__(self, annotations: pd.DataFrame):
        self.annotations = annotations
        self.image_paths = []
        self.all_boxes = []
        self.all_classes = []
        self.dataset = None

    def load_image_and_annotations(self, num_objects: int = 9, size_boxes: int = 4,
                                   num_classes: int = len(CLASS_TO_INDEX),
                                   skip_barcodes: bool = False, skip_trailer: bool = False) -> 'DatasetBuilder':
        grouped = self.annotations.groupby('filename')

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
                    break

                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                boxes[i] = [y_min, x_min, y_max, x_max]
                classes[i] = tf.one_hot(CLASS_TO_INDEX[str(row['class'])], depth=num_classes).numpy()

            self.image_paths.append(image_path)
            self.all_boxes.append(boxes)
            self.all_classes.append(classes)

        boxes_tensor = tf.convert_to_tensor(self.all_boxes, dtype=tf.float32)
        classes_tensor = tf.convert_to_tensor(self.all_classes, dtype=tf.float32)

        image_paths_ds = tf.data.Dataset.from_tensor_slices(self.image_paths)
        boxes_ds = tf.data.Dataset.from_tensor_slices(boxes_tensor)
        classes_ds = tf.data.Dataset.from_tensor_slices(classes_tensor)

        self.dataset = tf.data.Dataset.zip((image_paths_ds, boxes_ds, classes_ds))
        return self

    def load_and_process_image(image_path: str, target_size: int = TARGET_SIZE) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        original_shape = tf.shape(image)
        image = tf.image.resize_with_pad(image, target_size, target_size)
        image = image / 255.0
        assert image.shape == (target_size, target_size, 3)
        return image, original_shape

    def apply_augmentations(self, image_path: tf.Tensor, boxes: tf.Tensor, classes: tf.Tensor) -> tf.data.Dataset:
        original_image, original_shape = load_and_process_image(image_path.numpy().decode('utf-8'))
        adjusted_boxes = adjust_boxes_for_padding(boxes, original_shape, TARGET_SIZE)
        augmented_images = [original_image]

        for augmentation in self.augmentations:
            augmented_images.append(augmentation(original_image))

        return flatten_dataset(augmented_images, adjusted_boxes, classes)

    def add_augmentation(self, augmentation: Callable[[tf.Tensor], tf.Tensor]) -> 'DatasetBuilder':
        self.augmentations.append(augmentation)
        return self

    def build(self) -> tf.data.Dataset:
        return self.dataset

def flatten_dataset(images: List[tf.Tensor], boxes: tf.Tensor, classes: tf.Tensor) -> tf.data.Dataset:
    datasets = [tf.data.Dataset.from_tensors((image, boxes, classes)) for image in images]
    combined_ds = datasets[0]
    for ds in datasets[1:]:
        combined_ds = combined_ds.concatenate(ds)
    return combined_ds

def load_and_process_image(image_path: str, target_size: int = TARGET_SIZE) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    original_shape = tf.shape(image)
    image = tf.image.resize_with_pad(image, target_size, target_size)
    image = image / 255.0
    assert image.shape == (target_size, target_size, 3)
    return image, original_shape

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
import tensorflow as tf
from src.config.params import BATCH_SIZE

def get_dataset_size(dataset: tf.data.Dataset) -> int:
    """
    Returns the size of the dataset
    :param dataset:
    :return: the size of the dataset
    """
    size = 0
    for _ in dataset:
        size += 1
    return size


def split_dataset(dataset: tf.data.Dataset) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    total_size = sum(1 for _ in dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.2)
    test_size = total_size - train_size - val_size

    train_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    remaining_dataset = dataset.skip(train_size)

    val_dataset = remaining_dataset.take(val_size).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = remaining_dataset.skip(val_size).batch(BATCH_SIZE).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

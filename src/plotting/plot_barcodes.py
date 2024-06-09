import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import os
import tensorflow as tf

from src.config.params import IMAGE_DIRECTORY
from src.utils.dataset_tools import get_dataset_size


def plot_single_barcode_bounding_boxes(df: pd.DataFrame, filename: str) -> None:
    """
    Plots bounding boxes on a single image based on the provided DataFrame and filename.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box coordinates and class labels.
        filename (str): Name of the image file.
    """
    filtered_df = df[df['filename'] == filename]

    image_path = os.path.join(IMAGE_DIRECTORY, filename)
    image = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for _, row in filtered_df.iterrows():
        rect = patches.Rectangle(
            (row['xmin'], row['ymin']),
            row['xmax'] - row['xmin'],
            row['ymax'] - row['ymin'],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(row['xmax'], row['ymax'], row['class'], verticalalignment='top', color='blue')

    ax.set_xlim(0, filtered_df['width'].max())
    ax.set_ylim(filtered_df['height'].max(), 0)
    ax.set_title(filename)
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_images_with_bboxes_from_dataset_with_range(dataset: tf.data.Dataset, start_index: int, end_index: int) -> None:
    '''
    Plots images from a dataset
    :param dataset:
    :param start_index:
    :param end_index:
    :return:
    '''
    '''
    Nice but makes it extremely slow

    dataset_size = get_dataset_size(dataset)

    if start_index < 0 or end_index > dataset_size or start_index >= end_index:
        print("Invalid range. Adjusting to fit within the dataset.")
        start_index = max(0, min(start_index, dataset_size - 1))
        end_index = max(start_index + 1, min(end_index, dataset_size))
        raise ValueError("Invalid range. Adjusting to fit within the dataset")
    '''

    dataset = dataset.skip(start_index).take(end_index - start_index)

    for i, element in enumerate(dataset):
        image, boxes, classes = element

        image_np = image.numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        boxes_np = boxes.numpy()

        plt.imshow(image_np)

        for box in boxes_np:
            y_min, x_min, y_max, x_max = box * image_np.shape[0]
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red')
            plt.gca().add_patch(rect)

        plt.title(f'Image {start_index + i + 1}')
        plt.show()

def plot_single_digit(df: pd.DataFrame, filename: str) -> None:
    """
    Plots bounding boxes on a single image based on the provided DataFrame and filename.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box coordinates and class labels.
        filename (str): Name of the image file.
    """
    filtered_df = df[df['filename'] == filename]

    image_path = str('C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcode_Digits/' + filename)
    image = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    ax.set_xlim(0, filtered_df['width'].max())
    ax.set_ylim(filtered_df['height'].max(), 0)
    ax.set_title(filename)
    ax.axis('off')

    plt.tight_layout()
    plt.show()
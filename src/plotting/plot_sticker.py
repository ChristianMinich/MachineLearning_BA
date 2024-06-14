import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os


def plot_image_with_bboxes(image_path, bboxes):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for _, row in bboxes.iterrows():
        xmin, ymin, xmax, ymax, label = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['class']

        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        plt.text(xmin, ymin - 10, label, color='red', fontsize=12, weight='bold')

    # Show the plot
    plt.show()


def main(csv_path, image_directory, image_filename):
    df = pd.read_csv(csv_path)
    bboxes = df[df['filename'] == image_filename]
    image_path = os.path.join(image_directory, image_filename)
    plot_image_with_bboxes(image_path, bboxes)

if __name__ == '__main__':
    CSV_PATH = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcode_Sticker/adjusted_annotations.csv'
    IMAGE_DIRECTORY = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcode_Sticker/'
    IMAGE_FILENAME = 'IMG_0896_MOV-0017_jpg.rf.392b53456f49df2d5c1b2ab33d17f1c0.jpg'

    main(CSV_PATH, IMAGE_DIRECTORY, IMAGE_FILENAME)

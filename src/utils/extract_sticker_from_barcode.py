import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image_with_bboxes(image, bboxes):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for _, row in bboxes.iterrows():
        xmin, ymin, xmax, ymax, label = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['class']
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        plt.text(xmin, ymin - 10, label, color='red', fontsize=12, weight='bold')

    plt.show()


def process_barcodes(csv_path, image_directory, output_folder_path):
    df = pd.read_csv(csv_path)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    all_new_rows = []

    for image_filename in df['filename'].unique():
        image_df = df[df['filename'] == image_filename]
        aufkleber_row = image_df[image_df['class'] == 'Aufkleber']

        if aufkleber_row.empty:
            continue

        x_min_aufkleber = aufkleber_row['xmin'].values[0]
        y_min_aufkleber = aufkleber_row['ymin'].values[0]
        x_max_aufkleber = aufkleber_row['xmax'].values[0]
        y_max_aufkleber = aufkleber_row['ymax'].values[0]

        image_path = os.path.join(image_directory, image_filename)
        image = cv2.imread(image_path)

        if image is not None:
            print(f"Processing image: {image_filename}")
            print(f"Aufkleber bbox: ({x_min_aufkleber}, {y_min_aufkleber}), ({x_max_aufkleber}, {y_max_aufkleber})")

            cut_image = image[y_min_aufkleber:y_max_aufkleber, x_min_aufkleber:x_max_aufkleber]
            new_width, new_height = cut_image.shape[1], cut_image.shape[0]

            output_image_path = os.path.join(output_folder_path, image_filename)
            cv2.imwrite(output_image_path, cut_image)

            new_rows = []
            for _, row in image_df.iterrows():
                if row['class'] == 'Aufkleber':
                    continue

                # Adjust the bounding boxes relative to the cropped image
                x_min = row['xmin'] - x_min_aufkleber
                y_min = row['ymin'] - y_min_aufkleber
                x_max = row['xmax'] - x_min_aufkleber
                y_max = row['ymax'] - y_min_aufkleber

                print(f"Original bbox: ({row['xmin']}, {row['ymin']}), ({row['xmax']}, {row['ymax']})")
                print(f"Adjusted bbox: ({x_min}, {y_min}), ({x_max}, {y_max})")

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(new_width, x_max)
                y_max = min(new_height, y_max)

                new_rows.append([image_filename, new_width, new_height, row['class'], x_min, y_min, x_max, y_max])

            all_new_rows.extend(new_rows)

            '''
            plot_image_with_bboxes(cut_image, pd.DataFrame(new_rows,
                                                           columns=['filename', 'width', 'height', 'class', 'xmin',
                                                                   'ymin', 'xmax', 'ymax']))
            '''
        else:
            print(f"Warning: Image {image_filename} not found.")

    new_df = pd.DataFrame(all_new_rows,
                          columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    new_df.to_csv(os.path.join(output_folder_path, 'adjusted_annotations.csv'), index=False)


if __name__ == '__main__':
    CSV_PATH = r'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcodes_big/prepared_annotations.csv'
    IMAGE_DIRECTORY = r'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcodes_big/'
    OUTPUT_FOLDER_PATH = r'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcode_Sticker/'

    process_barcodes(CSV_PATH, IMAGE_DIRECTORY, OUTPUT_FOLDER_PATH)

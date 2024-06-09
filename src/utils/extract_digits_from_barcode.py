import pandas as pd
import os
import cv2

def extract_digits_from_barcode():
    CSV_PATH = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcodes_v10i/_annotations.csv'
    df = pd.read_csv(CSV_PATH)
    OUTPUT_FOLDER_PATH = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcode_Digits/'
    IMAGE_DIRECTORY = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcodes_v10i/'

    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    for index, row in df.iterrows():
        label = row['class']
        if label in ['Aufkleber', 'Strichcode']:
            continue
        x_min = row['xmin']
        y_min = row['ymin']
        x_max = row['xmax']
        y_max = row['ymax']
        image_filename = row['filename']

        image_path = os.path.join(IMAGE_DIRECTORY, image_filename)
        #print(image_path)
        image = cv2.imread(str(image_path))

        if image is not None:
            bounding_box = image[y_min:y_max, x_min:x_max]
            print('filename: ', image_filename, ' bounding box: ', bounding_box)

            output_path = os.path.join(OUTPUT_FOLDER_PATH, f"{image_filename}_{label}.jpg")

            cv2.imwrite(output_path, bounding_box)
        else:
            print(f"Warning: Image {image_filename} not found.")

if __name__ == '__main__':
    extract_digits_from_barcode()
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf

def load_digit_and_annotations():
    DIGIT_FOLDER_PATH = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcode_Digits/'
    images = []
    labels = []

    target_size = (224, 224)  # Set a target size for the images

    for filename in os.listdir(DIGIT_FOLDER_PATH):
        if filename.endswith('.jpg'):
            img_path = os.path.join(DIGIT_FOLDER_PATH, filename)
            #print(f"loading: {img_path}")  # Debugging print statement
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)  # Resize the image
            img_array = img / 255.0  # Normalize the image
            img_class = int(filename.rsplit('_', 1)[-1].split('.')[0])
            images.append(img_array)
            labels.append(img_class)

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    NUM_OF_DIGIT_CLASSES = 10

    labels = tf.one_hot(labels, depth=NUM_OF_DIGIT_CLASSES)

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

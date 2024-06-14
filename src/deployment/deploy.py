import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models


def prediction(model_path: str, image_path: str):
    try:
        model = models.load_model(model_path)

        model_input_shape = model.input_shape[1:3]

        image = Image.open(image_path).convert('RGB')

        image = image.resize(model_input_shape)

        image_array = np.array(image)

        image_tensor = np.expand_dims(image_array, axis=0)

        image_tensor = image_tensor.astype('float32')

        image_tensor /= 255.0

        predictions = model.predict(image_tensor)
        print(predictions)
        predicted_class = np.argmax(predictions)
        print(predicted_class)

        plot_image_with_class(image, predicted_class)

    except Exception as e:
        print(e)
        return None



def plot_image_with_class(image, predicted_class):
    fig = plt.figure(figsize=(3, 6))
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.savefig('prediction.png', dpi=fig.dpi)
    plt.show()


if __name__ == '__main__':
    model_path = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/models/epochs50_BS32_TS128_digit_classifier.keras'
    image_path = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/data/Barcode_Digits/IMG_2995_MOV-0012_jpg.rf.4e1135447449ae336160084096c53bd4.jpg_9.jpg'
    prediction(model_path, image_path)

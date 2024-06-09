import os

# DATA
CSV_FILE = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'Barcodes_v10i', '_annotations.csv'))
IMAGE_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'Barcodes_v10i'))
CLASS_TO_INDEX = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                  '7': 7, '8': 8, '9': 9, 'Aufkleber': 10, 'Strichcode': 11}

# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# z.P
TARGET_SIZE = 128
FILENAME = f"epochs{EPOCHS}_BS{BATCH_SIZE}_TS{TARGET_SIZE}"
NUM_CLASSES = 12
LEN_NUMBER = 7
PREDICTION_THRESHOLD = 0.5

TARGET_SIZE4DIGIT = 64
TARGET_SIZE4DIGIT_BBOX = 512

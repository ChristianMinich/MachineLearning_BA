"""
IMAGE_DIRECTORY = '/cluster/user/thoadelt/ML_Datasets/Barcodes_big'
CSV_FILE = IMAGE_DIRECTORY + '/prepared_annotations.csv'
"""
IMAGE_DIRECTORY = r'/cluster/user/chminich/Barcodes_big/'
CSV_FILE = IMAGE_DIRECTORY + r'prepared_annotations.csv'

CLASS_TO_INDEX = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                  '7': 7, '8': 8, '9': 9, 'Aufkleber': 10, 'Strichcode': 11}

DIGIT_CLASS_TO_INDEX = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                        '7': 7, '8': 8, '9': 9}

EPOCHS = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 24
TARGET_SIZE = 512
TARGET_SIZE4DIGIT = 64
TARGET_SIZE4DIGIT_BBOX = 512
FILENAME = f"epochs{EPOCHS}_BS{BATCH_SIZE}_TS{TARGET_SIZE}"
NUM_CLASSES = 12
LEN_NUMBER = 7
PREDICTION_THRESHOLD = 0.5

SAVE_PATH = '/cluster/user/chminich/'
SAVE_PATH_MODEL = SAVE_PATH + 'models'
SAVE_PATH_HIST = SAVE_PATH + 'histories'
LOGDIR = SAVE_PATH + 'logs'

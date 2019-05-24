#GLOBAL MODEL SETTINGS

#Data set directories pathes

#GENERAL SETTINGS
DATA_SET_DIR = "../DataSet/"
TRAIN_DATA_DIR = DATA_SET_DIR + "train-data/"
TEST_DATA_DIR =  DATA_SET_DIR + "test-data/"
VALID_DATA_DIR = DATA_SET_DIR + "valid-data/"


SERIALIZED_TRAIN_DATA_SAVE_LOCSTION = DATA_SET_DIR + "train_set_serialized_data/"
SERIALIZED_VALID_DATA_SAVE_LOCSTION = DATA_SET_DIR + "valid_set_serialized_data/"

BOTTLE_NECK_FEATURES_SAVE_LOC = "bottle_neck_features/"

IMAGE_SIZE = 64
CHANNELS = 3

EPOCHS = 5
BATCH_SIZE = 64

# Classoification categories list
CATEGORIES = ["Baby Bananas", "boat", "brie cheese", "Cavendish Bananas", "feta cheese", "Gala Apple", "Golden Delicious Apple", "Gouda cheese", "Parmigian cheese"]


LEARNING_TRANSFER_TENSORBOARD_LOG_DIR = "tensorboard logs/"

#LEARNING_TRANSFER_FROM_VGG16_MODEL_TRAINNED_ON_IMGNET_DATASET TOP MODEL HYPER PARAMETERS SETTINGS


LEARNING_TRANSFARE_MODEL = 'learning-transfer-VGG16-9-classes-classification/'  

LERNING_TRANSFARE_KEEP_PROB = 1.0

LT_LEARNING_RATE = 0.1

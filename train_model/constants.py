IMAGE_SIZE = 224 # Default image size for use with MobileNetV2
BATCH_SIZE = 32 # Function to load and preprocess each image
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

IMAGE_MEAN = 127.5 # Needed as input for MobileNet
IMAGE_STD = 127.5

LEARNING_RATE = 0.0001
LEARNING_RATE_FINETUNE = LEARNING_RATE / 10

NUM_EPOCHS = 30
VAL_STEPS = 20

# Increase training epochs for fine-tuning
NUM_EPOCHS_FINETUNE = 30
NUM_EPOCHS_TOTAL =  NUM_EPOCHS + NUM_EPOCHS_FINETUNE

PATH_ART = './downloads/art/'
PATH_TRASH = './downloads/trash/'
PATH_MODEL = 'model.h5'
PATH_TFLITE_FLOAT = 'model.float.tflite'
PATH_TFLITE_QUANT = 'model.quant.tflite'

LOGDIR="logs/scalars/"

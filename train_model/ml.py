import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime


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

LOGDIR="logs/scalars/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)


def _parse_fn(filename, label=None):
    img_raw = tf.io.read_file(filename)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, (IMAGE_SIZE, IMAGE_SIZE))
    img_final = (img_final-IMAGE_MEAN)/IMAGE_STD
    return img_final, label


def get_data():
    files_art = [str(f) for f in glob.glob(PATH_ART + "*.JPG", recursive=False)]
    files_trash = [str(f) for f in glob.glob(PATH_TRASH + "*.JPG", recursive=False)]

    labels_art = [0 for _ in files_art]
    labels_trash = [1 for _ in files_trash]

    train_filenames, val_filenames, train_labels, val_labels = train_test_split(files_art + files_trash, labels_art + labels_trash, train_size=0.9, random_state=42)

    num_train = len(train_filenames)

    train_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(train_filenames), tf.constant(train_labels))
    )
    val_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(val_filenames), tf.constant(val_labels))
    )

    # Run _parse_fn over each example in train and val datasets
    # Also shuffle and create batches
    train_data = train_data.map(_parse_fn).shuffle(buffer_size=10000).batch(BATCH_SIZE)
    val_data = val_data.map(_parse_fn).shuffle(buffer_size=10000).batch(BATCH_SIZE)
    
    return train_data, val_data, num_train

def make_model():
    # Pre-trained model with MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the pre-trained model weights
    base_model.trainable = False
    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    # Layer classification head with feature detector
    model = tf.keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return base_model, model


def make_model_finetune(base_model, model):
    # Unfreeze all layers of MobileNetV2
    base_model.trainable = True

    # Refreeze layers until the layers we want to fine-tune
    for layer in base_model.layers[:100]:
        layer.trainable =  False
    # Recompile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE_FINETUNE),
        metrics=['accuracy']
    )
    return model


def training(model, train_data, val_data, num_train, finetune=False):
    steps_per_epoch = round(num_train)//BATCH_SIZE
    # Train the Model
    model.fit(
        train_data.repeat(),
        epochs=NUM_EPOCHS_TOTAL if finetune else NUM_EPOCHS,
        initial_epoch = NUM_EPOCHS if finetune else 0,
        steps_per_epoch = steps_per_epoch,
        validation_data=val_data.repeat(), 
        validation_steps=VAL_STEPS,
        callbacks=[tensorboard_callback]
    )
    return model


def save_model(model):
    tf.keras.models.save_model(
        model,
        PATH_MODEL,
        overwrite=True,
        include_optimizer=True
    )


if __name__ == "__main__":
    train_data, val_data, num_train = get_data()
    base_model, model = make_model()
    model = training(model, train_data, val_data, num_train=num_train)
    model = make_model_finetune(base_model, model)
    model = training(model, train_data, val_data, num_train=num_train, finetune=True)
    save_model(model)

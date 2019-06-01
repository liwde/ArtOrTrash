import tensorflow as tf


IMAGE_SIZE = 224 # Default image size for use with MobileNetV2


def _parse_fn(filename, label=None):
    img_raw = tf.io.read_file(filename)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, (IMAGE_SIZE, IMAGE_SIZE))
    img_final = img_final/255.0
    return img_final, label


model = tf.keras.models.load_model(
    'model.h5',
    custom_objects=None,
    compile=True
)

test_image, _ = _parse_fn('test.jpg')
test_image = tf.reshape(test_image, (-1, 224, 224, 3))
result = model.predict(test_image)
print('Das ist Kunst!' if round(result[0][0]) == 0 else 'Das kann weg!')

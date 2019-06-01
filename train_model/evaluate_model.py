import tensorflow as tf
from constants import PATH_MODEL
from util import load_image


model = tf.keras.models.load_model(PATH_MODEL)

test_image = load_image('test.jpg')
test_image = tf.reshape(test_image, (-1, 224, 224, 3))

result = model.predict(test_image)

print('Das ist Kunst!' if round(result[0][0]) == 0 else 'Das kann weg!')

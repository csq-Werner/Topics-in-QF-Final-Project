import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
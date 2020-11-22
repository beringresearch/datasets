import tensorflow as tf

def read_png_uint16(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3, dtype=tf.uint16)
    image = tf.cast(image, tf.uint16)
    return image

def read_png_uint8(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)
    image = tf.cast(image, tf.uint8)
    return image
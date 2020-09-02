import tensorflow as tf
import tensorflow_io as tfio


def read_dicom_uint16(path):
    image = tf.io.read_file(path)
    image = tfio.image.decode_dicom_image(image, color_dim=True,
                                          dtype=tf.uint16, scale='auto',
                                          on_error='strict')[0]
    image = tf.image.grayscale_to_rgb(tf.cast(image, tf.int32))
    image = tf.cast(image, tf.float32)
    
    return image

def read_dicom_uint8(path):
    image = tf.io.read_file(path)
    image = tfio.image.decode_dicom_image(image, color_dim=True,
                                          dtype=tf.uint8, scale='auto',
                                          on_error='lossy')[0]
    image = tf.image.grayscale_to_rgb(tf.cast(image, tf.int32))
    image = tf.cast(image, tf.float32)
    
    return image
import tensorflow as tf
import tensorflow_addons as tfa
from .random_eraser import get_random_eraser

""" Image augmentation functions supported.
    Inputs are expected to be of type uintx. 
    Output are expected to be of the same type as input. """

def random_contrast(x, lower=0.1, upper=5):
    x = tf.image.random_contrast(x, lower, upper)
    return x

def flip(x):
    x = tf.image.flip_left_right(x)
    return x

def random_brightness(x, max_delta=0.25):
    x = tf.image.random_brightness(x, max_delta=max_delta)
    return x

def random_gamma(x, lower=0.15, upper=2.75):
    x = tf.image.adjust_gamma(x, tf.random.uniform([], lower, upper))
    return x

def random_rotate(x, rotation_range=10):
    dtype = x.dtype
    x = tf.cast(x, tf.float32)
    angle = tf.random.uniform([], -rotation_range, rotation_range)
    x = tfa.image.rotate(x, angle)
    x = tf.cast(x, dtype)
    return x

def random_rotate90(x):
    rotate_left = tf.random.uniform([]) > 0.5
    if rotate_left:
        x = tf.image.rot90(x, 3)
    else:
        x = tf.image.rot90(x, 1)
    return x

def invert_image(x):
    if x.dtype is tf.uint8:
        x = 255 - x
    elif x.dtype is tf.uint16:
        x = 65535 - x
    return x

def random_center_crop(x):
    dtype = x.dtype
    img_shape = x.shape
    x = tf.image.central_crop(x, 0.90)
    x = tf.image.resize(x, img_shape[:2])
    return tf.cast(x, dtype)

def random_erase(x, p=1.0, pixel_level=True):
    dtype = x.dtype
    if dtype is tf.uint8:
        max_val = 255
    elif dtype is tf.uint16:
        max_val = 65535
    else:
        raise ValueError("Unsupported dtype {}".format(dtype))
    random_eraser = get_random_eraser(p=p, pixel_level=pixel_level, v_h=max_val)
    x = tf.py_function(func=random_eraser, inp=[tf.cast(x, tf.float32)], Tout=tf.float32)
    x = tf.cast(x, dtype)
    return x
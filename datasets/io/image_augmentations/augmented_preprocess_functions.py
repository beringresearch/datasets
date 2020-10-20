import tensorflow as tf
import numpy as np

# Stacking
def build_stacking_image_augmentation_function(augmentation_list):
    @tf.function
    def stacking_augmentation_function(x):
        for augmentation in augmentation_list:
            apply = tf.random.uniform([]) > 0.5
            if apply:
                x = augmentation(x)
        return x
    return stacking_augmentation_function

# Exclusive
def build_exclusive_image_augmentation_function(augmentation_list):
    @tf.function
    def exclusive_augmentation_function(x):
        list_length = len(augmentation_list)
        random_augmentation_idx = tf.random.uniform([], minval=0, maxval=list_length, dtype=tf.int32)
        for i, random_augmentation in enumerate(augmentation_list):
            if i == random_augmentation_idx:
                x = random_augmentation(x)
        return x
    return exclusive_augmentation_function
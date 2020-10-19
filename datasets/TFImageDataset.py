"""TFImageDataset class definition"""
import os

from functools import partial
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE

class ImagePreprocessArgs:
    def __init__(self, target_size=(299, 299),
                 dtype=tf.dtypes.uint8,
                 preserve_aspect_ratio=False,
                 color_mode='rgb', interpolation='nearest'):
        self.target_size = target_size
        self.dtype = dtype
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.color_mode = color_mode
        self.interpolation = interpolation


class TFImageDataset:
    """Generate batches of tensor image data with real-time data augmentation.
     The data will be looped over (in batches).

     # Arguments
         read_function function that is applied to read image data from disk.
         augmentation_function function that applies a custom augmentation
             schedule to original images
         preprocessing_function: function that will be applied on each input.
            The function will run after the image is resized and augmented.
        shard: Tuple. Shard images across multiple GPUs. Takes on the format
            of (size, local_rank)
        prefetch: Boolean. Prefetching overlaps the preprocessing and model
            execution of a training step. While the model is executing
            training step s, the input pipeline is reading the data for step
            s+1. Doing so reduces the step time to the maximum
            (as opposed to the sum) of the training and the time it takes to
            extract the data.
        prefetch_gpu: String. A transformation that prefetches dataset values
            to the given device. This is useful if you'd like to prefetch
            images directly to a GPU. Acceptible values - None or
            '/device:GPU:0' where 0 can be replaced with another valid
        GPU index.

     # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a NumPy array of image data
                (in the case of a single image input) or a list
                of NumPy arrays (in the case with
                additional inputs) and `y` is a NumPy array
                of corresponding labels.
    """

    def __init__(self,
                 read_function=None,
                 augmentation_function=None,
                 preprocessing_function=None, shard=None,
                 prefetch=True, prefetch_gpu=None):

        self.read_function = read_function
        self.augmentation_function = augmentation_function
        self.preprocessing_function = preprocessing_function
        self.prefetch = prefetch
        self.prefetch_gpu = prefetch_gpu
        self.shard = shard


    def flow(self, x, y=None, batch_size=32, shuffle=True, repeat=True, random_state=None):
        """Takes data & label arrays, generates batches of augmented data.

            # Arguments

                x: Input data. Numpy array of rank 4 or a tuple. If tuple, the first
                    element should contain the images and the second element another
                    numpy array or a list of numpy arrays that gets passed to the output
                    without any modifications. Can be used to feed the model miscellaneous
                    data along with the images. In case of grayscale data, the channels
                    axis of the image array should have value 1, in case of RGB data, it
                    should have value 3, and in case of RGBA data, it should have value 4.
                y: Labels.
                batch_size: Int (default: 32).
                shuffle: Boolean (default: True).
                random_state: Int (default: None).
                repeat: whether to repeat the data (default: True)
        """

        dataset = self.__create_dataset('numpy', x, y, shuffle=shuffle,
                                        batch_size=batch_size, repeat=repeat,
                                        random_state=random_state, image_args=None)

        return dataset


    def flow_from_dataframe(self, dataframe, directory=None, x_col='filename', y_col=None,
                            color_mode='rgb', class_mode='categorical', classes=None,
                            target_size=(299, 299), dtype=tf.dtypes.uint8,
                            preserve_aspect_ratio=False, batch_size=32,
                            shuffle=True, repeat=True, interpolation='nearest',
                            validate_filenames=True, random_state=None):

        """Takes the dataframe and the path to a directory
         and generates batches of augmented/normalized data.

            # Arguments
                dataframe: Pandas dataframe containing the filepaths relative to
                    `directory` (or absolute paths if `directory` is None) of the
                    images in a string column. It should include other column/s
                    depending on the `class_mode`:
                    - if `class_mode` is `"categorical"` (default value) it must
                    include the `y_col` column with the class/es of each image.
                    Values in column can be string/list/tuple if a single class
                    or list/tuple if multiple classes.
                directory: string, path to the directory to read images from. If `None`
                    data in `x_col` column should be absolute paths.
                x_col: string, column in `dataframe` that contains the filenames (or
                    absolute paths if `directory` is `None`).
                y_col: string or list, column/s in `dataframe` that has the target data.
                target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
                    The dimensions to which all images found will be resized.
                dtype: Image data type: tf.dtypes.uint8 or tf.dtypes.uint16
                color_mode: one of "grayscale", "rgb". Default: "rgb".
                    Whether the images will be converted to have 1 or 3 color channels.
                classes: optional list or dict of classes (e.g. `['dogs', 'cats']`).
                    Default: None. If not provided, the list of classes will be
                    automatically inferred from the `y_col`,
                    which will map to the label indices, will be alphanumeric).
                class_mode: one of "categorical", "raw", None. Mode for yielding the targets -
                 "categorical": 2D numpy array of one-hot encoded labels. "raw": numpy array
                 of values in y_col column(s). Suitable for regression.
				 None: no targets are returned.
                batch_size: size of the batches of data (default: 32).
                shuffle: whether to shuffle the data (default: True)
                repeat: whether to repeat the data (default: True)
                interpolation: Interpolation method used to resample the image if the
                    target size is different from that of the loaded image.
                    Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
                    By default, `"nearest"` is used.
                validate_filenames: Boolean. whether to validate file names on disk (default: True)
                random_state: integer. Control random seed.
        """

        image_args = ImagePreprocessArgs(target_size=target_size,
                                         color_mode=color_mode,
                                         preserve_aspect_ratio=preserve_aspect_ratio,
                                         interpolation=interpolation,
                                         dtype=dtype)

        if directory is not None:
            if isinstance(x_col, str):
                img_filepaths = [os.path.join(directory, f) for f in dataframe[x_col].values]
            elif isinstance(x_col, tuple) or isinstance(x_col, list):
                img_filepaths = tuple([os.path.join(directory, f) for f in dataframe[col]] for col in x_col)
        else:
            if isinstance(x_col, str):
                img_filepaths = dataframe[x_col].values
            elif isinstance(x_col, tuple) or isinstance(x_col, list):
                img_filepaths = tuple(dataframe[col] for col in x_col)

        if validate_filenames:
            if isinstance(x_col, str):
                exists = [os.path.exists(f) for f in img_filepaths]

            elif isinstance(x_col, tuple) or isinstance(x_col, list):
                exists = [os.path.exists(f) for col in x_col for f in dataframe[col]]

            n_missing = len(exists) - np.sum(exists)
            print("Missing images: " + str(n_missing))
            if n_missing > 0:
                raise ValueError('Identified missing images in dataframe')


        if class_mode == 'categorical':
            if isinstance(y_col, str):
                label_encodings = _label_encoding(dataframe[y_col].values, classes)
                label_encodings = to_categorical(label_encodings)
            else:
                label_encodings = {key: _label_encoding(dataframe[key], classes[key]) for key in list(classes.keys())}
                label_encodings = {key: to_categorical(label_encodings[key]) for key in list(label_encodings.keys())}

        if class_mode == 'raw':
            label_encodings = dataframe[y_col].values

        if class_mode is None:
            label_encodings = None

        dataset = self.__create_dataset('dataframe', img_filepaths, label_encodings, shuffle=shuffle,
                                        batch_size=batch_size, repeat=repeat,
                                        random_state=random_state, image_args=image_args)

        return dataset

    def __create_dataset(self, type, x, y, shuffle=True, batch_size=32, repeat=True,
                         random_state=None, image_args=None):

        n_rows = 1
        if isinstance(x, tuple):
            n_rows = len(x)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if self.shard is not None:
            dataset = dataset.shard(self.shard[0], self.shard[1])

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(y), seed=random_state)

        dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.repeat()

        if self.read_function is not None:
            @tf.function
            def read_fn(paths, label, image_args):
                flattened_paths = _flatten_multi_input(paths)
                images = tf.map_fn(self.read_function, flattened_paths,
                                   fn_output_signature=image_args.dtype)
                images = tf.map_fn(lambda x: _resize_image(x, image_args=image_args), images,
                                   fn_output_signature=image_args.dtype)
                return images, label
            dataset = dataset.map(partial(read_fn, image_args=image_args), num_parallel_calls=AUTOTUNE)
        else:
            if type == 'dataframe':
                read_fn = _build_multi_input_load_fn(image_args=image_args)
                dataset = dataset.map(partial(read_fn, image_args=image_args), num_parallel_calls=AUTOTUNE)
            if type == 'numpy':
                pass

        if self.augmentation_function is not None:
            def augment_fn(image, label):
                image = self.augmentation_function(image)
                return image, label

            dataset = dataset.map(augment_fn, num_parallel_calls=AUTOTUNE)


        if self.preprocessing_function is not None:
            def preprocess_fn(image, label):
                image = self.preprocessing_function(image)
                return image, label

            dataset = dataset.map(preprocess_fn, num_parallel_calls=AUTOTUNE)

        if self.prefetch:
            dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        if self.prefetch_gpu:
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device(self.prefetch_gpu))

        dataset = dataset.map(lambda x, y: (_unflatten_multi_input(x, n_rows), y))

        if n_rows > 1:
            dataset = dataset.map(lambda x, y: ({f"input_{i+1}": x[i] for i in range(n_rows)}, y),
                                  num_parallel_calls=AUTOTUNE)
        else:
            dataset = dataset.map(lambda x, y: (x[0], y), num_parallel_calls=AUTOTUNE)

        return dataset


def _build_multi_input_load_fn(image_args=None):

    if image_args.color_mode == 'rgb':
        channels = 3

    if image_args.color_mode == 'grayscale':
        channels = 1

    @tf.function
    def _load_image_from_path_label(paths, labels, image_args=None):
        paths_flattened = tf.reshape(paths, (-1,))
        images_bytes = tf.map_fn(tf.io.read_file, paths_flattened)
        images_decoded = tf.map_fn(lambda x: tf.io.decode_png(x, channels=channels),
                                   images_bytes, fn_output_signature=image_args.dtype)
        images_resized = tf.map_fn(lambda x: _resize_image(x, image_args), images_decoded,
                                   fn_output_signature=image_args.dtype)

        return images_resized, labels

    return _load_image_from_path_label

def _flatten_multi_input(paths):
    paths_flattened = tf.reshape(paths, (-1,))
    return paths_flattened

def _unflatten_multi_input(images, n_rows):
    img_shape = tf.shape(images)
    images_reshaped = tf.reshape(images,
                                 (n_rows, -1, img_shape[1], img_shape[2], img_shape[3]))
    return images_reshaped

"""def _build_multi_input_load_fn(paths, image_args=None):
    n_rows = len(paths)

    if image_args.color_mode == 'rgb':
        channels = 3

    if image_args.color_mode == 'grayscale':
        channels = 1

    @tf.function
    def _load_image_from_path_label(paths, labels, image_args=None):
        paths_flattened = tf.reshape(paths, (-1,))
        images_bytes = tf.map_fn(tf.io.read_file, paths_flattened)
        images_decoded = tf.map_fn(lambda x: tf.io.decode_png(x, channels=channels),
                                   images_bytes, fn_output_signature=image_args.dtype)
        img_shape = tf.shape(images_decoded)

        images_reshaped = tf.reshape(images_decoded,
                                     (n_rows, -1, img_shape[1], img_shape[2], img_shape[3]))
        images_resized = tf.map_fn(lambda x: _resize_image(x, image_args), images_reshaped,
                                   fn_output_signature=image_args.dtype)

        return {f"input_{i+1}": images_resized[i] for i in range(n_rows)}, labels

    return _load_image_from_path_label
"""

def _load_image_from_path_label(path, label, image_args=None):
    if image_args.color_mode == 'rgb':
        channels = 3

    if image_args.color_mode == 'grayscale':
        channels = 1

    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=channels, dtype=image_args.dtype)
    image = _resize_image(image, image_args)

    image = tf.cast(image, tf.float32)

    return image, label

def _resize_image(image, image_args=None):
    dtype = image.dtype
    if image_args.preserve_aspect_ratio:
        image = tf.image.resize_with_pad(image, *image_args.target_size,
                                         method=image_args.interpolation)
    else:
        image = tf.image.resize(image, image_args.target_size,
                                method=image_args.interpolation)
    image = tf.cast(image, dtype)

    return image

def _label_encoding(vector, classes=None):
    if classes is None:
        integer_mapping = {x: i for i, x in enumerate(vector)}
    else:
        integer_mapping = {x: i for i, x in enumerate(classes)}

    label_encoding = [integer_mapping[word] for word in vector]
    return label_encoding

"""TFImageDataset class definition"""
import os

from functools import partial
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ImagePreprocessArgs:
	def __init__(self, target_size=(299, 299),
				 preserve_aspect_ratio=False,
				 color_mode='rgb', interpolation='nearest'):
		self.target_size = target_size
		self.preserve_aspect_ratio = preserve_aspect_ratio,
		self.color_mode = color_mode
		self.interpolation = interpolation


class TFImageDataset:
	 """Generate batches of tensor image data with real-time data augmentation.
     The data will be looped over (in batches).

     # Arguments
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
        prefetch_gpu: Boolean. A transformation that prefetches dataset values
        	to the given device. This is useful if you'd like to prefetch
        	images directly to a GPU.

     # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a NumPy array of image data
                (in the case of a single image input) or a list
                of NumPy arrays (in the case with
                additional inputs) and `y` is a NumPy array
                of corresponding labels.
     """

	def __init__(self, augmentation_function=None,
				 preprocessing_function=None, shard=None,
				 prefetch=True, prefetch_gpu=False):

		self.augmentation_function = augmentation_function
		self.preprocessing_function = preprocessing_function
		self.prefetch = prefetch
		self.prefetch_gpu = prefetch_gpu
		self.shard = shard

	def flow_from_dataframe(self, dataframe, directory='.', x_col='filename', y_col='class',
							color_mode='rgb', class_mode='categorical', classes=None,
							target_size=(299, 299), preserve_aspect_ratio=False, batch_size=32,
							shuffle=True, repeat=False, interpolation='nearest', 
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
	            color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb".
	                Whether the images will be converted to have 1 or 3 color channels.
	            classes: optional list of classes (e.g. `['dogs', 'cats']`).
	                Default: None. If not provided, the list of classes will be
	                automatically inferred from the `y_col`,
	                which will map to the label indices, will be alphanumeric).
	                The dictionary containing the mapping from class names to class
	                indices can be obtained via the attribute `class_indices`.
	            class_mode: one of "categorical"
	            batch_size: size of the batches of data (default: 32).
	            shuffle: whether to shuffle the data (default: True)
	            seed: optional random seed for shuffling and transformations.
	            interpolation: Interpolation method used to resample the image if the
	                target size is different from that of the loaded image.
	                Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
	                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
	                supported. If PIL version 3.4.0 or newer is installed, `"box"` and
	                `"hamming"` are also supported. By default, `"nearest"` is used.
	            validate_filenames: Boolean, whether to validate image filenames in
	                `x_col`. If `True`, invalid images will be ignored. Disabling this
	                option can lead to speed-up in the execution of this function.
	                Default: `True`.
	        # Returns
	            A `Dataset` yielding tuples of `(x, y)`
	            where `x` is a NumPy array containing a batch
	            of images with shape `(batch_size, *target_size, channels)`
	            and `y` is a NumPy array of corresponding labels.
	        """

		image_args = ImagePreprocessArgs(target_size=target_size, color_mode=color_mode,
										 preserve_aspect_ratio=preserve_aspect_ratio,
										 interpolation=interpolation)


		img_filepaths = [os.path.join(directory, file) for file in dataframe[x_col].values]

		if validate_filenames:
			exists = [os.path.exists(f) for f in img_filepaths]
			n_missing = len(exists) - np.sum(exists)
			print("Missing images: " + str(n_missing))

			if n_missing > 0:
				raise ValueError('Identified missing images in dataframe')


		if class_mode == 'categorical':
			label_encodings = _label_encoding(dataframe[y_col].values, classes)
			label_encodings = to_categorical(label_encodings)

		dataset = tf.data.Dataset.from_tensor_slices((img_filepaths, label_encodings))

		if self.shard is not None:
			dataset = dataset.shard(self.shard[0], self.shard[1])

		if shuffle:
			dataset = dataset.shuffle(buffer_size=len(img_filepaths), seed=random_state)

		if repeat:
			dataset = dataset.repeat()

		if self.augmentation_function is not None:
			augment_fn = self.augmentation_function
			dataset = dataset.map(augment_fn)

		dataset = dataset.map(partial(
						_load_and_preprocess_from_path_label,
						preprocessing_function=self.preprocessing_function,
						image_args=image_args), num_parallel_calls=AUTOTUNE)

		dataset = dataset.batch(batch_size)

		if self.prefetch:
			dataset = dataset.prefetch(buffer_size=AUTOTUNE)
		if self.prefetch_gpu:
			dataset = dataset.apply(tf.data.experimental.prefetch_to_device(self.prefetch_gpu))

		return dataset


def _load_and_preprocess_from_path_label(path, label,
										 preprocessing_function=None,
										 image_args=None):

	return _load_and_preprocess_image(path=path,
    								  preprocessing_function=preprocessing_function,
    								  image_args=image_args), label

def _load_and_preprocess_image(path,
							   preprocessing_function=None,
							   image_args=None):
    image = tf.io.read_file(path)
    dicom = tf.strings.split(path, '.')[-1] == 'dcm'
    if not dicom:
        return _preprocess_image(image, preprocessing_function, image_args)
    return _preprocess_dicom(image, preprocessing_function, image_args)

def _preprocess_dicom(image_bytes,
                     preprocessing_function=None,
                     image_args=None
                     ):

    image = tfio.image.decode_dicom_image(image_bytes,
                                          color_dim=True, dtype=tf.uint8,
                                          scale='auto',
                                          on_error='lossy')[0]
    image = tf.image.grayscale_to_rgb(image)
    if image_args.preserve_aspect_ratio:
        image = tf.image.resize_with_pad(image, *image_args.target_size,
                                         method=image_args.interpolation)
    else:
        image = tf.image.resize(image, image_args.target_size,
                                method=image_args.interpolation)
    image = tf.cast(image, tf.float32)
    if preprocessing_function is not None:
        image = preprocessing_function(image)
    return image


def _preprocess_image(image, preprocessing_function=None, image_args=None):

	if image_args.color_mode == 'rgb':
		channels = 3

	if image_args.color_mode == 'grayscale':
		channels = 1

	image = tf.image.decode_png(image, channels=channels)
	if image_args.preserve_aspect_ratio:
		image = tf.image.resize_with_pad(image, *image_args.target_size, method=image_args.interpolation)
	else:
		image = tf.image.resize(image, image_args.target_size, method=image_args.interpolation)
	image = tf.cast(image, tf.float32)
	if preprocessing_function is not None:
		image = preprocessing_function(image)
	return image

def _label_encoding(vector, classes=None):
	if classes is None:
		integer_mapping = {x: i for i, x in enumerate(vector)}
	else:
		integer_mapping = {x: i for i, x in enumerate(classes)}

	label_encoding = [integer_mapping[word] for word in vector]
	return label_encoding

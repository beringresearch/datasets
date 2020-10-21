"""TFImageDataset class definition"""
import os

from functools import partial
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import pandas as pd

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ImagePreprocessArgs:
	def __init__(self, target_size=(299, 299),
				 dtype=tf.dtypes.uint8,
				 preserve_aspect_ratio=False,
				 color_mode='rgb', interpolation='nearest'):
		self.target_size = target_size
		self.dtype = dtype
		self.preserve_aspect_ratio = preserve_aspect_ratio,
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

	def __init__(self,
				 read_function=None,
				 augmentation_function=None,
				 preprocessing_function=None, shard=None,
				 prefetch=True, prefetch_gpu=False):

		self.read_function=read_function
		self.augmentation_function = augmentation_function
		self.preprocessing_function = preprocessing_function
		self.prefetch = prefetch
		self.prefetch_gpu = prefetch_gpu
		self.shard = shard

	def flow(self, x, y=None, batch_size=32, shuffle=True, repeat=False, random_state=None):
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
				repeat: whether to repeat the data (default: False)
		"""

		dataset = self.__create_dataset('numpy', x, y, shuffle=shuffle,
										batch_size=batch_size, repeat=repeat,
										random_state=random_state, image_args=None)

		return dataset



	def flow_from_dataframe(self, dataframe, directory=None, x_col='filename', y_col='class',
							color_mode='rgb', class_mode='categorical', classes=None,
							target_size=(299, 299), dtype=tf.dtypes.uint8,
							preserve_aspect_ratio=False, batch_size=32,
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
	            dtype: Image data type: tf.dtypes.uint8 or tf.dtypes.uint16
	            color_mode: one of "grayscale", "rgb". Default: "rgb".
	                Whether the images will be converted to have 1 or 3 color channels.
	            classes: optional list of classes (e.g. `['dogs', 'cats']`).
	                Default: None. If not provided, the list of classes will be
	                automatically inferred from the `y_col`,
	                which will map to the label indices, will be alphanumeric).
	                The dictionary containing the mapping from class names to class
	                indices can be obtained via the attribute `class_indices`.
	            class_mode: one of "categorical", "raw", None. Mode for yielding the targets -
	             "categorical": 2D numpy array of one-hot encoded labels. "raw": numpy array
	             of values in y_col column(s). Suitable for regression. None: no targets are returned.
	            batch_size: size of the batches of data (default: 32).
	            shuffle: whether to shuffle the data (default: True)
	            repeat: whether to repeat the data (default: False)
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

		multiinput = False
		if type(x_col) is list:
			multiinput = True

		if directory is not None:
			if multiinput:
				img_filepaths = {}
				for feature in x_col:
					img_filepaths[feature] = directory + os.sep + dataframe[feature].values
				img_filepaths = pd.DataFrame(img_filepaths)
			else:
				img_filepaths = dataframe[x_col].apply(lambda row: os.path.join(directory, row))
		else:
			img_filepaths = dataframe[x_col]

		if validate_filenames:
			exists = [os.path.exists(f) for f in np.unique(img_filepaths.values.ravel('K'))]
			n_missing = len(exists) - np.sum(exists)
			print("Missing images: " + str(n_missing))

			if n_missing > 0:
				raise ValueError('Identified missing images in dataframe')


		if class_mode == 'categorical':
			label_encodings = _label_encoding(dataframe[y_col].values, classes)
			label_encodings = to_categorical(label_encodings)

		if class_mode == 'raw':
			label_encodings = dataframe[y_col].values

		if class_mode is None:
			label_encodings = None

		dataset = self.__create_dataset('dataframe', img_filepaths, label_encodings, multiinput=multiinput,
										shuffle=shuffle,
										batch_size=batch_size, repeat=repeat,
										random_state=random_state, image_args=image_args)

		return dataset

	def __create_dataset(self, type, x, y, multiinput=False, shuffle=True, batch_size=32, repeat=False,
						 random_state=None, image_args=None):

		if multiinput:
			inputs = x.to_dict(orient='list')
		else:
			inputs = list(x.values)

		dataset = tf.data.Dataset.from_tensor_slices((inputs, y))

		if self.shard is not None:
			dataset = dataset.shard(self.shard[0], self.shard[1])

		if shuffle:
			dataset = dataset.shuffle(buffer_size=len(x), seed=random_state)

		if repeat:
			dataset = dataset.repeat()

		if self.read_function is not None:
			def read_fn(path, label):
				image = self.read_function(path)
				return image, label
			dataset = dataset.map(read_fn, num_parallel_calls=AUTOTUNE)
		else:
			if type == 'dataframe':
				read_fn = _load_image_from_path_label
				dataset = dataset.map(partial(read_fn, image_args=image_args), num_parallel_calls=AUTOTUNE)
			if type == 'numpy':
				pass

		dataset = dataset.map(partial(_resize_image, image_args=image_args), num_parallel_calls=AUTOTUNE)

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

		if multiinput:
			dataset = dataset.map(partial(_to_dictionary, keys=list(inputs.keys())), num_parallel_calls=AUTOTUNE)


		dataset = dataset.batch(batch_size)

		if self.prefetch:
			dataset = dataset.prefetch(buffer_size=AUTOTUNE)
		if self.prefetch_gpu:
			dataset = dataset.apply(tf.data.experimental.prefetch_to_device(self.prefetch_gpu))


		return dataset

def _to_dictionary(images, label, keys=None):
	output = ({keys[i]: images[i] for i in range(len(keys))}, label)
	return output


def _load_image_from_path_label(path, label, image_args=None):
	if image_args.color_mode == 'rgb':
		channels = 3

	if image_args.color_mode == 'grayscale':
		channels = 1

	image = tf.io.read_file(path)
	image = tf.io.decode_image(image, channels=channels)

	return image, label

def _resize_image(image, label, image_args=None):
	if image_args.preserve_aspect_ratio:
		image = tf.image.resize_with_pad(image, *image_args.target_size,
										 method=image_args.interpolation)
	else:
		image = tf.image.resize(image, image_args.target_size,
								method=image_args.interpolation)

	return image, label

def _label_encoding(vector, classes=None):
	if classes is None:
		integer_mapping = {x: i for i, x in enumerate(vector)}
	else:
		integer_mapping = {x: i for i, x in enumerate(classes)}

	label_encoding = [integer_mapping[word] for word in vector]
	return label_encoding

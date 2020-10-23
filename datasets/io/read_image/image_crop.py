import tensorflow as tf
import numpy as np


def read_image_from_filepath(filepath, **kwargs):
    """Default implementation of image read function"""
    img_bytes = tf.io.read_file(filepath)
    img = tf.image.decode_png(img_bytes, **kwargs)
    return img

def read_mask_from_filepath(filepath, **kwargs):
    """Default implementation of mask read function.
    Returns 3D image tensor with 1 color channel (shape: (x, y, 1))."""
    img_bytes = tf.io.read_file(filepath)
    img = tf.image.decode_png(img_bytes, channels=1, **kwargs)
    return img

def read_and_crop_image_mask(paths, read_img_fn=read_image_from_filepath,
                             read_mask_fn=read_mask_from_filepath, crop_shape=(299, 299),
                             crop_masked_img=True, soft_mask=True):
    """Reads an image and a mask from disk using the provided read functions
    and returns a randomly selected crop of the requested size that is centered around a
    segmented pixel.

    The mask is assumed to be scaled within the usual range for the provided dtype (e.g.
    0-255 for tf.uint8).

    Args:
        paths: list/dict. The first value is assumed to be the image and the second a mask.
        read_img_fn: function(x). Function to read image from disk.
        read_mask_fn: function(x). Function to read mask from disk.
        crop_shape: tuple(2). Requested crop size.
        crop_masked_img: boolean. If true, crops masked image rather than original.
        soft_mask: boolean. If true, the soft image mask is applied to the image,
            rather than a hard one.
    """

    if isinstance(paths, dict):
        paths = paths.values()

    img, mask = (read_fn(path) for read_fn, path in zip((read_img_fn, read_mask_fn), paths))
    mask /= mask.dtype.max
    mask_hard = tf.cast(tf.math.round(mask), dtype=img.dtype)

    if len(tf.shape(mask_hard)) < 3:
        mask_hard = tf.expand_dims(mask_hard, axis=-1)

    if crop_masked_img:
        if soft_mask:
            img_to_crop = tf.cast(tf.cast(img, tf.float32) * mask, img.dtype)
        else:
            img_to_crop = img * mask_hard
    else:
        img_to_crop = img

    crop_center_coords = select_random_segmented_pixel(mask_hard)
    cropped_img = centered_img_crop(img_to_crop, crop_center_coords, crop_shape)
    return cropped_img


def centered_img_crop(img, center_pixel_coords, crop_shape):
    """Extracts a crop from an centered around the coordinates provided
    in center_pixel_coords.

    If the requested crop would extend beyond the edge of the image it will be constrained
    to fit inside.

    Args:
        img - image Tensor
        center_pixel_coords: tuple/list (height, width) with coordinates to center of crop
        crop_shape: tuple/list of (height, width) with dimensions of the crop to take

    Returns:
        img_crop: Tensor
    """

    crop_shape = tf.cast(crop_shape, tf.int64)

    offset_height = max(center_pixel_coords[0] - crop_shape[0] // 2, 0)
    offset_height = min(offset_height, img.shape[0] - crop_shape[0])
    offset_width = max(center_pixel_coords[1] - crop_shape[1] // 2, 0)
    offset_width = min(offset_width, img.shape[1] - crop_shape[1])

    img_crop = tf.image.crop_to_bounding_box(
        img, offset_height=offset_height, offset_width=offset_width,
        target_height=crop_shape[0], target_width=crop_shape[1])

    return img_crop


def select_random_segmented_pixel(mask):
    """Takes a binary mask and outputs the coordinates of a random positive point.

    Args:
        mask - 2D Tensor or 3D where one dimension is size 1. 
        Assumes binary values [0, 1] where 1 denotes the candidates for sampling.
    """
    mask = tf.squeeze(mask)
    positive_segmentation_tensor = np.array([[1 for i in range(tf.shape(mask)[0])]
                                             for j in range(tf.shape(mask)[1])])
    segmented_pixel_coords = tf.where(tf.equal(mask, positive_segmentation_tensor))

    random_coord_idx = tf.random.uniform([], maxval=tf.shape(segmented_pixel_coords)[0], dtype=tf.int32)
    random_segmented_coord = segmented_pixel_coords[random_coord_idx]
    return random_segmented_coord


def draw_center_crop_box(img, center_pixel_coords, crop_shape):
    """Draws a bounding box centered around the coordinates provided
    in center_pixel_coords.

    If the requested crop would extend beyond the edge of the image it will be constrained
    to fit inside.

    Args:
        img - image Tensor
        center_pixel_coords: tuple/list (height, width) with coordinates to center of crop
        crop_shape: tuple/list of (height, width) with dimensions of the crop to take

    Returns:
        img_crop: Tensor
    """

    if len(img.shape) < 4:
        img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32) / img.dtype.max

    y_min = max(center_pixel_coords[0] - crop_shape[0] // 2, 0)
    y_min = min(y_min, img.shape[1] - crop_shape[0])
    y_max = y_min + crop_shape[0]
    y_min /= img.shape[1]
    y_max /= img.shape[1]

    x_min = max(center_pixel_coords[1] - crop_shape[1] // 2, 0)
    x_min = min(x_min, img.shape[2] - crop_shape[1])
    x_max = x_min + crop_shape[1]
    x_min /= img.shape[2]
    x_max /= img.shape[2]

    box = np.array([y_min, x_min, y_max, x_max])
    boxes = box.reshape([1, 1, 4])

    colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    return tf.image.draw_bounding_boxes(img, boxes, colors)

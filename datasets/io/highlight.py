import tensorflow.keras.backend as K

def highlight(path, model_name, size, layer_name, class_ix=None):
    """function to display the activation map of an image
    path: path to image file
    model_name: model name associated with activation map to be displayed
    size: image size required for model
    layer_name: layer associated with activation map
    class_ix: index of class to display. if None, use class 0
    """
    x = read_dicom_uint16(path, 1)
    x = tf.image.resize(image, (size, size), method='bilinear', preserve_aspect_ratio=False)
    x = tf.cast(x, tf.float32)
    x /= 32767.5
    x -= 1
    x = tf.reshape(x, (1, size, size, 3))
    
    with tf.GradientTape() as tape:
        last_conv_layer = model_name.get_layer(layer_name)
        iterate = tf.keras.models.Model([model_name.inputs], [model_name.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        if class_ix_highlight is not None:
            class_out = model_out[:, class_ix_highlight]
        else:
            class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)

        if np.max(grads) == 0:
            grads = grads + 1e-9
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.absolute(heatmap)
    heatmap /= np.max(heatmap)
    heatmap = np.squeeze(heatmap)

    original = tf.io.read_file(path)
    original = tfio.image.decode_dicom_image(original, dtype=tf.uint16)
    heatmap = tf.image.resize(heatmap.reshape((1, heatmap.shape[0], heatmap.shape[1], 1)), size=[original.shape[1], original.shape[2]])
    
    return original, heatmap
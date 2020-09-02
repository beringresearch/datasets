from tensorflow.keras.applications.inception_v3 import preprocess_input

def preprocess_uint16_input_inceptionv3(x):
    x /= 32767.5
    x -= 1
    return x

def preprocess_uint8_input_inceptionv3(x):
    return preprocess_input(x)
# Build Tensorflow Input Pipelines

`datasets` is a python package that enables users to quickly build complex Tensorflow datasets. The tool offers flexibility to import out-of-memory datasets and apply image augmentation functions in real time.

`datasets` API borrows heavily from [`ImageDataGenerator`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator), making it nearly a drop-in replacement. However, `TFImageDataset` class is approximately 5-fold faster than the `ImageDataGenerator`.

## Installation

The latest stable version can be installed directly from github:

```bash
git clone https://github.com/beringresearch/datasets/
cd datasets
python3 install --editable .
```

## Getting Started

Check out [example notebook](https://github.com/beringresearch/datasets/blob/master/examples/TFImageDataset.ipynb) to get started with the package.
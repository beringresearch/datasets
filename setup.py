"""Setup bering-datasets"""
from setuptools import setup

setup(name='datasets',
      version='0.2',
      description='Dataset management classes for Tensorflow',
      url='http://github.com/beringresearch/datasets',
      author='Bering Limited',
      license='Apache 2.0',
      packages=['datasets'],
      install_requires=['tensorflow',
                        'tensorflow-io'],
      zip_safe=False)

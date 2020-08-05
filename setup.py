"""Setup bering-datasets"""
from setuptools import setup

setup(name='datasets',
      version='0.1',
      description='Dataset management classes for Tensorflow',
      url='http://github.com/beringresearch/datasets',
      author='Bering Limited',
      license='Apache 2.0',
      packages=['datasets'],
      install_requires=['tensorflow<=2.2.0',
                        'tensorflow-io==0.14.0'],
      zip_safe=False)

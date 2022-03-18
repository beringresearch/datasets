"""Setup bering-datasets"""
from setuptools import setup, find_packages

setup(name='datasets',
      version='0.3.2',
      description='Dataset management classes for Tensorflow',
      url='http://github.com/beringresearch/datasets',
      author='Bering Limited',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=['tensorflow',
                        'tensorflow-io', 
                        'ipywidgets'],
      zip_safe=False)

#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='GAN',
      version='0.0.1',
      description='Generative Adversarial Nets.',
      author='Tomoshgie Shimomura',
      author_email='simomura@iu.nitech.ac.jp',
      packages=find_packages(),
      install_requires=['numpy', 'pillow', 'tensorflow', 'keras',
                        'matplotlib', 'seaborn', 'h5py'],
      entry_points={
          'console_scripts': [
              'gan = gan.gan:main',
          ],
      })

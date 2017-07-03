#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='GAN',
      version='0.2.1',
      description='Generative Adversarial Nets utility.',
      author='Tomoshgie Shimomura',
      author_email='simomura@iu.nitech.ac.jp',
      packages=find_packages(),
      install_requires=['numpy', 'pillow', 'keras',
                        'matplotlib', 'seaborn', 'h5py'],
      entry_points={
          'console_scripts': [
              'ganutil = gan.gan:main',
          ],
      })

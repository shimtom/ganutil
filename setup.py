#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='ganutil',
      version='0.2.3',
      description='Generative Adversarial Nets utility.',
      author='shimtom',
      author_email='ii00zero1230@gmail.com',
      packages=find_packages(),
      install_requires=['numpy', 'pillow', 'keras',
                        'matplotlib', 'seaborn', 'h5py'],
      entry_points={
          'console_scripts': [
              'ganutil = ganutil.ganutil:main',
          ],
      })

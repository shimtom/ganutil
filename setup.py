#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(name='ganutil',
      version='0.4.1',
      description='Generative Adversarial Nets utility.',
      author='shimtom',
      author_email='ii00zero1230@gmail.com',
      packages=find_packages(exclude=('test')),
      install_requires=['numpy', 'pillow', 'keras >= 2.0.6',
                        'matplotlib', 'seaborn', 'h5py'],
      entry_points={
          'console_scripts': [
              'ganutil = ganutil.ganutil:main',
          ],
      },
      test_suite='test')

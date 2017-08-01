#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(name='ganutil',
      version='0.4.1',
      description='Generative Adversarial Nets utility.',
      author='shimtom',
      author_email='ii00zero1230@gmail.com',
      packages=find_packages(),
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      install_requires=['numpy', 'pillow', 'keras >= 2.0.6',
                        'matplotlib', 'seaborn', 'h5py'],
      test_suite='test')

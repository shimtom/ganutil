import json
import os
import tempfile
from unittest import TestCase

import ganutil.utils as utils
import yaml
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD


class TestUtils(TestCase):
    def test_compile(self):
        discriminator = Sequential()
        discriminator.add(Dense(1, input_shape=(16,)))
        dparameter = {
            'optimizer': SGD(),
            'loss': binary_crossentropy
        }
        generator = Sequential()
        generator.add(Dense(16, input_shape=(32,)))
        gparameter = {
            'optimizer': SGD(),
            'loss': binary_crossentropy
        }
        gan, discriminator, generator = utils.compile(
            discriminator, generator, dparameter, gparameter)

        self.assertTrue(gan.built)
        self.assertTrue(discriminator.built)
        self.assertFalse(generator.built)

    def test_save_architecture(self):
        with tempfile.TemporaryDirectory() as dirpath:
            discriminator = Sequential()
            discriminator.add(Dense(1, input_shape=(16,)))
            generator = Sequential()
            generator.add(Dense(16, input_shape=(32,)))

            dpath = os.path.join(dirpath, 'discriminator')
            gpath = os.path.join(dirpath, 'generator')

            for de in ['.yml', '.json', '']:
                for ge in ['.yml', '.json', '']:
                    if de == '' or ge == '':
                        with self.assertRaises(ValueError):
                            utils.save_architecture(
                                dpath + de, gpath + ge, discriminator, generator)
                    else:
                        utils.save_architecture(
                            dpath + de, gpath + ge, discriminator, generator)
                        with open(dpath + de) as d:
                            try:
                                if de == '.yml':
                                    yaml.load(d)
                                elif de == '.json':
                                    json.load(d)
                                else:
                                    raise(ValueError('Unknown `de`: ' + str(de)))
                            except Exception as e:
                                self.fail(e)
                        with open(gpath + ge) as g:
                            try:
                                if ge == '.yml':
                                    yaml.load(g)
                                elif ge == '.json':
                                    json.load(g)
                                else:
                                    raise(ValueError('Unknown `ge`: ' + str(ge)))
                            except Exception as e:
                                self.fail(e)

    def test_set_trainability(self):
        pass

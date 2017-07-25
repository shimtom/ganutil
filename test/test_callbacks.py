import os
import tempfile
from unittest import TestCase

import ganutil.callbacks as cbks
import numpy as np
from keras.layers import Activation, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
import inspect

class TestCallbacks(TestCase):
    def test_inheritance(self):
        callbacks = [cbks.GeneratedImage, cbks.ValueGraph, cbks.LossGraph,
                     cbks.AccuracyGraph, cbks.ValueHistory, cbks.LossHistory,
                     cbks.AccuracyHistory, cbks.ProgbarLogger,
                     cbks.GanModelCheckpoint]

        for callback in callbacks:
            functions = ['on_batch_begin', 'on_batch_end', 'on_epoch_begin',
                         'on_epoch_end', 'on_train_begin', 'on_train_end',
                         'set_model', 'set_params']
            self.assertTrue(inspect.isclass(callback))
            members = inspect.getmembers(callback)
            for member in members:
                function = member[0]
                for i, f in enumerate(functions):
                    if f == function:
                        functions.pop(i)
                        break
            self.assertEqual(len(functions), 0)


    def test_generated_image(self):
        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, '{epoch:02d}/images.png')

            generator = Sequential()
            generator.add(Dense(16, input_shape=(32,)))
            generator.add(Activation('tanh'))
            generator.add(Reshape((4, 4, 1)))

            discriminator = Sequential()
            discriminator.add(Flatten(input_shape=(4, 4, 1)))
            discriminator.add(Dense(1))
            discriminator.add(Activation('sigmoid'))

            gan = Sequential((generator, discriminator))
            gan.compile(Adam(), 'binary_crossentropy')

            samples = np.random.uniform(-1, -1, (25, 32))

            def normalize(images):
                return images * 127.5 + 127.5

            callback = cbks.GeneratedImage(filepath, samples, normalize)

            callback.set_model(gan)
            callback.set_params({
                'epochs': 10,
                'steps': 1,
                'verbose': 1,
                'do_validation': False,
                'metrics': ['loss'],
            })

            for i in range(10):
                callback.on_epoch_end(i,logs={})
                self.assertTrue(os.path.isfile(filepath.format(epoch=i)))

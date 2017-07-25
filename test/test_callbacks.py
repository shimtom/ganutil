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

    def test_value_graph(self):
        epoch_logs = {'value': 10}
        batch_logs = {'size': 10, 'value': 1}
        name = 'value'

        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, '{epoch:02d}/graph.png')

            callback = cbks.ValueGraph(filepath, name, sample_mode='epoch')
            callback.on_train_begin()
            for i in range(10):
                callback.on_epoch_begin(i, epoch_logs)
                for j in range(10):
                    callback.on_batch_begin(j, batch_logs)
                    callback.on_batch_end(j, batch_logs)
                    self.assertFalse(os.path.isfile(filepath.format(epoch=i, batch=j)))
                callback.on_epoch_end(i, epoch_logs)
                self.assertTrue(os.path.isfile(filepath.format(epoch=i, batch=-1)))
            callback.on_train_end()

        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, '{batch:02d}/graph.png')

            callback = cbks.ValueGraph(filepath, name, sample_mode='batch')
            callback.on_train_begin()
            for i in range(10):
                callback.on_epoch_begin(i, epoch_logs)
                for j in range(10):
                    callback.on_batch_begin(j, batch_logs)
                    callback.on_batch_end(j, batch_logs)
                    self.assertTrue(os.path.isfile(filepath.format(epoch=i, batch=j)))
                callback.on_epoch_end(i, epoch_logs)
                self.assertFalse(os.path.isfile(filepath.format(epoch=i, batch=-1)))
            callback.on_train_end()

class TestGeneratedImage(TestCase):
    """ganutil.callbacks.GeneratedImageをテストするクラス."""
    def test_property(self):
        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, '{epoch:02d}/images.png')

            samples = np.random.uniform(-1, -1, (25, 32))

            def normalize(images):
                return images * 127.5 + 127.5

            callback = cbks.GeneratedImage(filepath, samples, normalize)

            self.assertEqual(callback.filepath, filepath)
            self.assertTrue(np.array_equal(callback.samples, samples))
            self.assertEqual(callback.normalize, normalize)

    def test_callback(self):
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

            for i in range(10):
                callback.on_epoch_end(i,logs={})
                self.assertTrue(os.path.isfile(filepath.format(epoch=i)))

class TestValueGraph(TestCase):
    def test_callback(self):
        epoch_logs = {'value': 10}
        batch_logs = {'size': 10, 'value': 1}
        name = 'value'

        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, '{epoch:02d}/graph.png')

            callback = cbks.ValueGraph(filepath, name, sample_mode='epoch')
            callback.on_train_begin()
            for i in range(10):
                callback.on_epoch_begin(i, epoch_logs)
                for j in range(10):
                    callback.on_batch_begin(j, batch_logs)
                    callback.on_batch_end(j, batch_logs)
                    self.assertFalse(os.path.isfile(filepath.format(epoch=i, batch=j)))
                callback.on_epoch_end(i, epoch_logs)
                self.assertTrue(os.path.isfile(filepath.format(epoch=i, batch=-1)))
            callback.on_train_end()

        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, '{batch:02d}/graph.png')

            callback = cbks.ValueGraph(filepath, name, sample_mode='batch')
            callback.on_train_begin()
            for i in range(10):
                callback.on_epoch_begin(i, epoch_logs)
                for j in range(10):
                    callback.on_batch_begin(j, batch_logs)
                    callback.on_batch_end(j, batch_logs)
                    self.assertTrue(os.path.isfile(filepath.format(epoch=i, batch=j)))
                callback.on_epoch_end(i, epoch_logs)
                self.assertFalse(os.path.isfile(filepath.format(epoch=i, batch=-1)))
            callback.on_train_end()

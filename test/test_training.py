import math
import unittest
from unittest import TestCase, skip

import ganutil.callbacks as cbks
import numpy as np
import pytest
import tensorflow as tf
from ganutil import fit_generator
from keras.callbacks import Callback, ModelCheckpoint
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Flatten,
                          Reshape)
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence


class DCallback(Callback):
    def __init__(this):
        super(DCallback, this).__init__()
        this.count_on_train_begin = 0
        this.count_on_train_end = 0
        this.count_on_epoch_begin = 0
        this.count_on_epoch_end = 0
        this.count_on_batch_begin = 0
        this.count_on_batch_end = 0

    def on_train_begin(this, logs={}):
        this.count_on_train_begin += 1
        assert logs == {}

    def on_train_end(this, logs={}):
        this.count_on_train_end += 1

    def on_epoch_begin(this, epoch, logs={}):
        this.count_on_epoch_begin += 1
        assert logs == {}

    def on_epoch_end(this, epoch, logs={}):
        this.count_on_epoch_end += 1
        assert 'loss' in logs
        assert 'acc' in logs

    def on_batch_begin(this, batch, logs={}):
        this.count_on_batch_begin += 1
        assert 'size' in logs

    def on_batch_end(this, batch, logs={}):
        this.count_on_batch_end += 1
        assert 'loss' in logs
        assert 'acc' in logs


class GCallback(Callback):
    def __init__(this):
        super(GCallback, this).__init__()
        this.count_on_train_begin = 0
        this.count_on_train_end = 0
        this.count_on_epoch_begin = 0
        this.count_on_epoch_end = 0
        this.count_on_batch_begin = 0
        this.count_on_batch_end = 0

    def on_train_begin(this, logs={}):
        this.count_on_train_begin += 1
        assert logs == {}

    def on_train_end(this, logs={}):
        this.count_on_train_end += 1

    def on_epoch_begin(this, epoch, logs={}):
        this.count_on_epoch_begin += 1
        assert logs == {}

    def on_epoch_end(this, epoch, logs={}):
        this.count_on_epoch_end += 1
        assert 'loss' in logs
        assert 'acc' in logs

    def on_batch_begin(this, batch, logs={}):
        this.count_on_batch_begin += 1
        assert 'size' in logs

    def on_batch_end(this, batch, logs={}):
        this.count_on_batch_end += 1
        assert 'loss' in logs
        assert 'acc' in logs


def create_test_gan():
    generator = Sequential()
    generator.add(Dense(16, input_shape=(32,)))
    generator.add(Activation('tanh'))
    generator.add(Reshape((4, 4, 1)))
    generator = generator
    generator._make_predict_function()
    generator_graph = tf.get_default_graph()

    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=(4, 4, 1)))
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))
    discriminator = discriminator
    discriminator.compile(
        Adam(), 'binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    gan = Sequential((generator, discriminator))
    gan = gan
    gan.compile(Adam(), 'binary_crossentropy', metrics=['accuracy'])

    return gan, generator, discriminator, generator_graph


class TestFitGenerator(TestCase):
    """ganutil.fit_generator()のテストを行うクラス."""
    @skip
    def test_fit_generator(self):
        """ganutil.fit_generatorでganが訓練されることを確認する"""
        def discriminator_model():
            model = Sequential()
            model.add(Conv2D(64, (5, 5), strides=(2, 2),
                             padding='same', input_shape=(28, 28, 1)))
            model.add(Activation('tanh'))
            model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
            model.add(Activation('tanh'))
            model.add(Flatten())
            model.add(Dense(1024))
            model.add(Activation('tanh'))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            return model

        def generator_model():
            model = Sequential()
            model.add(Dense(1024, input_dim=100))
            model.add(Activation('tanh'))
            model.add(Dense(128 * 7 * 7))
            model.add(BatchNormalization())
            model.add(Activation('tanh'))
            model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
            model.add(Conv2DTranspose(
                64, (5, 5), strides=(2, 2), padding='same'))
            model.add(Activation('tanh'))
            model.add(Conv2DTranspose(
                1, (5, 5), strides=(2, 2), padding='same'))
            model.add(Activation('tanh'))
            model._make_predict_function()
            graph = tf.get_default_graph()
            return model, graph
        generator, generator_graph = generator_model()

        discriminator = discriminator_model()
        discriminator.compile(Adam(lr=0.0005),
                              'binary_crossentropy', metrics=['accuracy'])
        discriminator.trainable = False
        gan = Sequential((generator, discriminator))

        gan.compile(Adam(lr=0.0005), 'binary_crossentropy',
                    metrics=['accuracy'])

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        mnist_data = np.array(
            X_train, dtype=np.float32).reshape((-1, 28, 28, 1))
        mnist_data = (mnist_data - 127.5) / 127.5
        data_size = len(mnist_data)

        def d_generator(batch_size):
            while True:
                indices = np.random.permutation(data_size)
                for i in range(0, math.ceil(data_size / batch_size), batch_size):
                    with generator_graph.as_default():
                        ginputs = np.random.uniform(-1, 1, [batch_size, 100])
                        ginputs = ginputs.astype(np.float32)
                        inputs = generator.predict_on_batch(ginputs)
                        targets = np.zeros(len(inputs), dtype=np.int64)
                        yield inputs, targets

                    inputs = mnist_data[indices[i:i + batch_size]]
                    targets = np.ones(len(inputs))
                    yield inputs, targets

        def g_generator(batch_size):
            while True:
                inputs = np.random.uniform(-1, 1, [batch_size, 100])
                inputs = inputs.astype(np.float32)
                targets = np.ones(len(inputs))
                yield inputs, targets

        d_callbacks = [cbks.LossGraph('./save/{epoch:02d}/dloss.png'),
                       cbks.AccuracyGraph('./save/{epoch:02d}/dacc.png'),
                       cbks.LossHistory('./save/{epoch:02d}/dloss.npy'),
                       cbks.AccuracyHistory('./save/{epoch:02d}/dacc.npy'),
                       ModelCheckpoint('./save/discriminator.h5')
                       ]
        samples = np.random.uniform(-1, 1, [25, 100]).astype(np.float32)
        samples[0] = -1.
        samples[-1] = 1.
        samples[len(samples) // 2] = 0

        def normalize(images):
            return np.array(images) * 127.5 + 127.5

        g_callbacks = [cbks.LossGraph('./save/{epoch:02d}/gloss.png'),
                       cbks.AccuracyGraph('./save/{epoch:02d}/gacc.png'),
                       cbks.LossHistory('./save/{epoch:02d}/gloss.npy'),
                       cbks.AccuracyHistory('./save/{epoch:02d}/gacc.npy'),
                       cbks.GeneratedImage('./save/{epoch:02d}/images.png',
                                           samples,
                                           normalize),
                       cbks.GanModelCheckpoint('./save/generator.h5')]

        epochs = 20
        batch_size = 128
        steps_per_epoch = math.ceil(data_size / batch_size)
        fit_generator(gan, discriminator, generator, d_generator(batch_size),
                      g_generator(batch_size), steps_per_epoch, epochs=epochs,
                      d_callbacks=d_callbacks, g_callbacks=g_callbacks,
                      initial_epoch=0)

    def test_callbacks(self):
        """コールバックが正しい回数呼び出されることを確認する."""
        gan, generator, discriminator, generator_graph = create_test_gan()

        def d_generator():
            while True:
                with generator_graph.as_default():
                    ginputs = np.random.uniform(-1,
                                                1, [5, 32]).astype(np.float32)
                    inputs = generator.predict_on_batch(ginputs)
                    targets = np.zeros(len(inputs), dtype=np.int)
                    yield inputs, targets

                inputs = np.zeros([5, 4, 4, 1])
                targets = np.ones(len(inputs))
                yield inputs, targets

        def g_generator():
            while True:
                inputs = np.random.uniform(-1, 1, [5, 32]).astype(np.float32)
                targets = np.ones(len(inputs))
                yield inputs, targets

        d_callback = DCallback()
        g_callback = GCallback()
        epochs = 5
        steps_per_epoch = 10
        d_iteration_per_step = 3
        g_iteration_per_step = 2
        fit_generator(gan, discriminator, generator,
                      d_generator(), g_generator(), steps_per_epoch,
                      d_iteration_per_step=d_iteration_per_step,
                      g_iteration_per_step=g_iteration_per_step,
                      epochs=epochs, d_callbacks=[d_callback],
                      g_callbacks=[g_callback])

        self.assertEqual(d_callback.count_on_train_begin, 1)
        self.assertEqual(d_callback.count_on_train_end, 1)
        self.assertEqual(d_callback.count_on_epoch_begin, epochs)
        self.assertEqual(d_callback.count_on_epoch_end, epochs)
        self.assertEqual(d_callback.count_on_batch_begin,
                         d_iteration_per_step * steps_per_epoch * epochs)
        self.assertEqual(d_callback.count_on_batch_end,
                         d_iteration_per_step * steps_per_epoch * epochs)

        self.assertEqual(g_callback.count_on_train_begin, 1)
        self.assertEqual(g_callback.count_on_train_end, 1)
        self.assertEqual(g_callback.count_on_epoch_begin, epochs)
        self.assertEqual(g_callback.count_on_epoch_end, epochs)
        self.assertEqual(g_callback.count_on_batch_begin,
                         g_iteration_per_step * steps_per_epoch * epochs)
        self.assertEqual(g_callback.count_on_batch_end,
                         g_iteration_per_step * steps_per_epoch * epochs)

    @skip
    def test_multi_processing(self):
        """データのジェネレートをマルチプロセスで動作できることを確認"""
        gan, generator, discriminator, generator_graph = create_test_gan()
        class DSequence(Sequence):
            def __init__(this, batch_size, data_size):
                this.batch_size = batch_size
                this.data_size = data_size

            def __getitem__(this, index):
                """Gets batch at position `index`.
                # Arguments
                    index: position of the batch in the Sequence.
                # Returns
                    A batch
                """
                if index % 2 == 0:
                    with generator_graph.as_default():
                        ginputs = np.random.uniform(-1, 1,
                                                    [this.batch_size, 32]).astype(np.float32)
                        inputs = generator.predict_on_batch(ginputs)
                        targets = np.zeros(len(inputs), dtype=np.int)
                else:
                    inputs = np.zeros([this.batch_size, 4, 4, 1])
                    targets = np.ones(len(inputs))
                return inputs, targets

            def __len__(this):
                """Number of batch in the Sequence.
                # Returns
                    The number of batches in the Sequence.
                """
                return this.data_size

        class GSequence(Sequence):
            def __init__(this, batch_size, data_size):
                this.batch_size = batch_size
                this.data_size = data_size

            def __getitem__(this, index):
                """Gets batch at position `index`.
                # Arguments
                    index: position of the batch in the Sequence.
                # Returns
                    A batch
                """
                inputs = np.random.uniform(-1, 1,
                                           [this.batch_size, 32]).astype(np.float32)
                targets = np.ones(len(inputs))
                return inputs, targets

            def __len__(this):
                """Number of batch in the Sequence.
                # Returns
                    The number of batches in the Sequence.
                """
                return this.data_size

        def d_generator():
            while True:
                with generator_graph.as_default():
                    ginputs = np.random.uniform(-1,
                                                1, [5, 32]).astype(np.float32)
                    inputs = generator.predict_on_batch(ginputs)
                    targets = np.zeros(len(inputs), dtype=np.int)
                    yield inputs, targets

                inputs = np.zeros([5, 4, 4, 1])
                targets = np.ones(len(inputs))
                yield inputs, targets

        def g_generator():
            while True:
                inputs = np.random.uniform(-1, 1, [5, 32]).astype(np.float32)
                targets = np.ones(len(inputs))
                yield inputs, targets

        d_callback = DCallback()
        g_callback = GCallback()
        epochs = 5
        steps_per_epoch = 10
        d_iteration_per_step = 3
        g_iteration_per_step = 2
        fit_generator(gan, discriminator, generator,
                      d_generator(), g_generator(), steps_per_epoch,
                      d_iteration_per_step=d_iteration_per_step,
                      g_iteration_per_step=g_iteration_per_step,
                      epochs=epochs, d_callbacks=[d_callback],
                      g_callbacks=[g_callback], max_queue_size=10, workers=1,
                      use_multiprocessing=True, initial_epoch=0)

        self.assertEqual(d_callback.count_on_train_begin, 1)
        self.assertEqual(d_callback.count_on_train_end, 1)
        self.assertEqual(d_callback.count_on_epoch_begin, epochs)
        self.assertEqual(d_callback.count_on_epoch_end, epochs)
        self.assertEqual(d_callback.count_on_batch_begin,
                         d_iteration_per_step * steps_per_epoch * epochs)
        self.assertEqual(d_callback.count_on_batch_end,
                         d_iteration_per_step * steps_per_epoch * epochs)

        self.assertEqual(g_callback.count_on_train_begin, 1)
        self.assertEqual(g_callback.count_on_train_end, 1)
        self.assertEqual(g_callback.count_on_epoch_begin, epochs)
        self.assertEqual(g_callback.count_on_epoch_end, epochs)
        self.assertEqual(g_callback.count_on_batch_begin,
                         g_iteration_per_step * steps_per_epoch * epochs)
        self.assertEqual(g_callback.count_on_batch_end,
                         g_iteration_per_step * steps_per_epoch * epochs)

        d_callback = DCallback()
        g_callback = GCallback()
        epochs = 5
        steps_per_epoch = 10
        d_iteration_per_step = 3
        g_iteration_per_step = 2
        fit_generator(gan, discriminator, generator,
                      DSequence(5, 30), GSequence(5, 20), steps_per_epoch,
                      d_iteration_per_step=d_iteration_per_step,
                      g_iteration_per_step=g_iteration_per_step,
                      epochs=epochs, d_callbacks=[d_callback],
                      g_callbacks=[g_callback], max_queue_size=10, workers=4,
                      use_multiprocessing=True, initial_epoch=0)

        self.assertEqual(d_callback.count_on_train_begin, 1)
        self.assertEqual(d_callback.count_on_train_end, 1)
        self.assertEqual(d_callback.count_on_epoch_begin, epochs)
        self.assertEqual(d_callback.count_on_epoch_end, epochs)
        self.assertEqual(d_callback.count_on_batch_begin,
                         d_iteration_per_step * steps_per_epoch * epochs)
        self.assertEqual(d_callback.count_on_batch_end,
                         d_iteration_per_step * steps_per_epoch * epochs)

        self.assertEqual(g_callback.count_on_train_begin, 1)
        self.assertEqual(g_callback.count_on_train_end, 1)
        self.assertEqual(g_callback.count_on_epoch_begin, epochs)
        self.assertEqual(g_callback.count_on_epoch_end, epochs)
        self.assertEqual(g_callback.count_on_batch_begin,
                         g_iteration_per_step * steps_per_epoch * epochs)
        self.assertEqual(g_callback.count_on_batch_end,
                         g_iteration_per_step * steps_per_epoch * epochs)


if __name__ == '__main__':
    unittest.main()

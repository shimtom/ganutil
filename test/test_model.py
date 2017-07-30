import math
import os
import tempfile

import ganutil.callbacks as cbks
import numpy as np
import pytest
from ganutil import Gan
from keras.callbacks import Callback, ModelCheckpoint
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Flatten,
                          Reshape)
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam


@pytest.fixture()
def dirpath():
    tempdir = tempfile.TemporaryDirectory()
    print('Directory: %s' % tempdir.name)
    yield tempdir.name
    print('Directory clean upped.')
    tempdir.cleanup()


@pytest.fixture()
def mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    mnist_data = np.array(X_train, dtype=np.float32)
    mnist_data = mnist_data.reshape((-1, 28, 28, 1))
    mnist_data = (mnist_data - 127.5) / 127.5
    return mnist_data


@pytest.fixture()
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


@pytest.fixture()
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

    return model


@pytest.fixture()
def compiled_gan(discriminator_model, generator_model):
    gan = Gan(generator_model, discriminator_model)

    doptimizer = Adam(lr=0.0005)
    goptimizer = Adam(lr=0.0005)
    dloss = 'binary_crossentropy'
    gloss = 'binary_crossentropy'
    dmetrics = ['accuracy']
    gmetrics = ['accuracy']

    gan.compile(doptimizer, goptimizer, dloss, gloss,
                dmetrics=dmetrics, gmetrics=gmetrics)

    return gan


@pytest.fixture()
def trainable_gan(compiled_gan, mnist_dataset):
    data_size = len(mnist_dataset)
    generator = compiled_gan.generator_model
    graph = compiled_gan.generator_graph

    def d_generator(batch_size):
        while True:
            indices = np.random.permutation(data_size)
            for i in range(0, math.ceil(data_size / batch_size), batch_size):
                with graph.as_default():
                    ginputs = np.random.uniform(-1, 1, [batch_size, 100])
                    ginputs = ginputs.astype(np.float32)
                    inputs = generator.predict_on_batch(ginputs)
                    targets = np.zeros(len(inputs), dtype=np.int64)
                    yield inputs, targets

                inputs = mnist_dataset[indices[i:i + batch_size]]
                targets = np.ones(len(inputs))
                yield inputs, targets

    def g_generator(batch_size):
        while True:
            inputs = np.random.uniform(-1, 1, [batch_size, 100])
            inputs = inputs.astype(np.float32)
            targets = np.ones(len(inputs))
            yield inputs, targets

    # check d_generator, g_generator
    dgen = d_generator(10)
    for _ in range(10):
        dgened = next(dgen)
        assert len(dgened) == 2
        assert np.array_equal(dgened[0].shape, [10, 28, 28, 1])
        assert np.array_equal(dgened[1].shape, [10, ])

    ggen = g_generator(10)
    for _ in range(10):
        ggened = next(ggen)
        assert len(ggened) == 2
        assert np.array_equal(ggened[0].shape, [10, 100])
        assert np.array_equal(ggened[1].shape, [10, ])

    return compiled_gan, d_generator, g_generator


@pytest.mark.base
@pytest.mark.parametrize("dmetrics, gmetrics", [
    (None, None),
    (None, ['accuracy']),
    (['accuracy'], None),
    (['accuracy'], ['accuracy']),
])
def test_compiled_gan(discriminator_model, generator_model, dmetrics, gmetrics):
    """コンパイルが正しく行われることを確認."""
    gan = Gan(discriminator_model, generator_model)

    doptimizer = Adam(lr=0.0005)
    goptimizer = Adam(lr=0.0005)
    dloss = 'binary_crossentropy'
    gloss = 'binary_crossentropy'
    dmetrics = None
    gmetrics = None
    dmetrics = ['accuracy']
    gmetrics = ['accuracy']

    gan.compile(doptimizer, goptimizer, dloss, gloss,
                dmetrics=dmetrics, gmetrics=gmetrics)
    assert gan.discriminator.built
    assert gan.discriminator.optimizer == doptimizer
    assert gan.discriminator.loss == dloss
    assert gan.discriminator.metrics == ([] or dmetrics)
    assert len(gan.generator.layers) == 2
    assert gan.generator.built
    assert gan.generator.optimizer == goptimizer
    assert gan.generator.loss == gloss
    assert gan.generator.metrics == ([] or gmetrics)
    assert len(gan.generator.layers[0].layers) == len(generator_model.layers)


@pytest.mark.callbacks
@pytest.mark.parametrize("epochs, steps_per_epoch, d_iter,  g_iter", [
    (1, 1, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
    (1, 2, 1, 1),
    (1, 2, 2, 1),
    (1, 2, 1, 2),
])
def test_callbacks(trainable_gan, dirpath, epochs, steps_per_epoch, d_iter,  g_iter):
    """訓練中にコールバックが正しく動作することを確認."""
    class Checker(Callback):
        def __init__(self):
            super(Checker, self).__init__()
            self.count_on_train_begin = 0
            self.count_on_train_end = 0
            self.count_on_epoch_begin = 0
            self.count_on_epoch_end = 0
            self.count_on_batch_begin = 0
            self.count_on_batch_end = 0

        def on_train_begin(self, logs={}):
            self.count_on_train_begin += 1
            assert logs == {}

        def on_train_end(self, logs={}):
            self.count_on_train_end += 1

        def on_epoch_begin(self, epoch, logs={}):
            self.count_on_epoch_begin += 1
            assert logs == {}

        def on_epoch_end(self, epoch, logs={}):
            self.count_on_epoch_end += 1
            assert 'loss' in logs
            assert 'acc' in logs

        def on_batch_begin(self, batch, logs={}):
            self.count_on_batch_begin += 1
            assert 'size' in logs

        def on_batch_end(self, batch, logs={}):
            self.count_on_batch_end += 1
            assert 'loss' in logs
            assert 'acc' in logs

    d_lossgraphpath = os.path.join(dirpath, 'save/{epoch:02d}/dloss.png')
    d_accgraphpath = os.path.join(dirpath, 'save/{epoch:02d}/dacc.png')
    d_losshistorypath = os.path.join(dirpath, 'save/{epoch:02d}/dloss.npy')
    d_acchistorypath = os.path.join(dirpath, 'save/{epoch:02d}/dacc.npy')
    d_modelpath = os.path.join(dirpath, 'save/generator.h5')

    d_checker = Checker()

    d_callbacks = [
        d_checker,
        cbks.LossGraph(d_lossgraphpath),
        cbks.AccuracyGraph(d_accgraphpath),
        cbks.LossHistory(d_losshistorypath),
        cbks.AccuracyHistory(d_acchistorypath),
        ModelCheckpoint(d_modelpath)
    ]

    samples = np.random.uniform(-1, 1, [25, 100]).astype(np.float32)
    samples[0] = -1.
    samples[-1] = 1.
    samples[len(samples) // 2] = 0

    def normalize(images):
        return np.array(images) * 127.5 + 127.5

    g_lossgraphpath = os.path.join(dirpath, 'save/{epoch:02d}/gloss.png')
    g_accgraphpath = os.path.join(dirpath, 'save/{epoch:02d}/gacc.png')
    g_losshistorypath = os.path.join(dirpath, 'save/{epoch:02d}/gloss.npy')
    g_acchistorypath = os.path.join(dirpath, 'save/{epoch:02d}/gacc.npy')
    g_imagepath = os.path.join(dirpath, 'save/{epoch:02d}/images.png')
    g_modelpath = os.path.join(dirpath, 'save/generator.h5')

    g_checker = Checker()
    g_callbacks = [
        g_checker,
        cbks.LossGraph(g_lossgraphpath),
        cbks.AccuracyGraph(g_accgraphpath),
        cbks.LossHistory(g_losshistorypath),
        cbks.AccuracyHistory(g_acchistorypath),
        cbks.GeneratedImage(g_imagepath, samples, normalize),
        cbks.GanModelCheckpoint(g_modelpath)]

    gan, d_generator, g_generator = trainable_gan

    gan.fit_generator(d_generator(10), g_generator(10), steps_per_epoch,
                      d_iteration_per_step=d_iter, g_iteration_per_step=g_iter,
                      d_callbacks=d_callbacks,
                      g_callbacks=g_callbacks,
                      epochs=1)

    assert d_checker.count_on_train_begin == 1
    assert d_checker.count_on_train_end == 1
    assert d_checker.count_on_epoch_begin == epochs
    assert d_checker.count_on_epoch_end == epochs
    assert d_checker.count_on_batch_begin == epochs * steps_per_epoch * d_iter
    assert d_checker.count_on_batch_end == epochs * steps_per_epoch * d_iter

    assert g_checker.count_on_train_begin == 1
    assert g_checker.count_on_train_end == 1
    assert g_checker.count_on_epoch_begin == epochs
    assert g_checker.count_on_epoch_end == epochs
    assert g_checker.count_on_batch_begin == epochs * steps_per_epoch * g_iter
    assert g_checker.count_on_batch_end == epochs * steps_per_epoch * g_iter

    for epoch in range(epochs):
        assert os.path.isfile(d_lossgraphpath.format(epoch=epoch))
        assert os.path.isfile(d_accgraphpath.format(epoch=epoch))
        assert os.path.isfile(d_losshistorypath.format(epoch=epoch))
        assert os.path.isfile(d_acchistorypath.format(epoch=epoch))

        assert os.path.isfile(g_lossgraphpath.format(epoch=epoch))
        assert os.path.isfile(g_accgraphpath.format(epoch=epoch))
        assert os.path.isfile(g_losshistorypath.format(epoch=epoch))
        assert os.path.isfile(g_acchistorypath.format(epoch=epoch))
        assert os.path.isfile(g_imagepath.format(epoch=epoch))

    assert os.path.isfile(d_modelpath)
    assert os.path.isfile(g_modelpath)


@pytest.mark.Train
@pytest.mark.parametrize("d_iter, g_iter, expected", [
    (0, 1, True),
    (1, 0, False),
])
def test_trained_separately(trainable_gan, d_iter, g_iter, expected):
    """discriminatorとgeneratorが別々に訓練されていることを確認."""
    gan, d_generator, g_generator = trainable_gan

    dbefore_weights = gan.discriminator.get_weights()
    gbefore_weights = gan.generator_model.get_weights()
    gan.fit_generator(d_generator, g_generator, 100,
                      d_iteration_per_step=d_iter, g_iteration_per_step=g_iter,
                      epochs=1)
    dafter_weights = gan.discriminator.get_weights()
    gafter_weights = gan.generator_model.get_weights()

    assert len(dbefore_weights) == len(dafter_weights)
    for w1, w2 in zip(dbefore_weights, dafter_weights):
        assert np.array_equal(w1, w2) == expected

    assert len(gbefore_weights) == len(gafter_weights)
    for w1, w2 in zip(gbefore_weights, gafter_weights):
        assert np.array_equal(w1, w2) != expected

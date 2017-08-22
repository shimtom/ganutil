import inspect
import os
import tempfile

import ganutil.callbacks as cbks
import numpy as np
import pytest
from keras.layers import Activation, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam


@pytest.fixture()
def dirpath():
    tempdir = tempfile.TemporaryDirectory()
    print('Directory: %s' % tempdir.name)
    yield tempdir.name
    print('Directory clean upped.')
    tempdir.cleanup()


@pytest.mark.callback
@pytest.mark.parametrize("callback", [
    cbks.GeneratedImage,
    cbks.ValueGraph,
    cbks.LossGraph,
    cbks.AccuracyGraph,
    cbks.ValueHistory,
    cbks.LossHistory,
    cbks.AccuracyHistory,
    cbks.GanProgbarLogger,
    cbks.GanModelCheckpoint
])
def test_inheritance(callback):
    functions = ['on_batch_begin', 'on_batch_end', 'on_epoch_begin',
                 'on_epoch_end', 'on_train_begin', 'on_train_end',
                 'set_model', 'set_params']

    assert inspect.isclass(callback)

    members = inspect.getmembers(callback)
    for member in members:
        function = member[0]
        for i, f in enumerate(functions):
            if f == function:
                functions.pop(i)
                break
    assert len(functions) == 0


@pytest.mark.callback
class TestGeneratedImage(object):
    """ganutil.callbacks.GeneratedImageをテストするクラス."""

    def test_property(self, dirpath):
        filepath = os.path.join(dirpath, '{epoch:02d}/images.png')

        samples = np.random.uniform(-1, -1, (25, 32))

        def normalize(images):
            return images * 127.5 + 127.5

        callback = cbks.GeneratedImage(filepath, samples, normalize)

        assert callback.filepath == filepath
        assert np.array_equal(callback.samples, samples)
        assert callback.normalize == normalize

    def test_callback(self, dirpath):
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
            callback.on_epoch_end(i, logs={})
            assert os.path.isfile(filepath.format(epoch=i))


@pytest.mark.callback
@pytest.mark.parametrize("filepath, sample_mode, epoch_mode", [
    ('{epoch:02d}/graph.png', 'epoch', True),
    ('{batch:02d}/graph.png', 'batch', False),
])
def test_valuegraph(dirpath, filepath, sample_mode, epoch_mode):
    epoch_logs = {'value': 10}
    batch_logs = {'size': 10, 'value': 1}
    name = 'value'

    filepath = os.path.join(dirpath, filepath)

    callback = cbks.ValueGraph(filepath, name, sample_mode=sample_mode)
    assert callback.filepath == filepath
    assert callback.name == name
    assert callback.epoch_mode == epoch_mode

    callback.on_train_begin()
    assert callback.values == []
    assert callback.epoch_values == []

    for i in range(10):
        callback.on_epoch_begin(i, epoch_logs)
        for j in range(10):
            callback.on_batch_begin(j, batch_logs)
            callback.on_batch_end(j, batch_logs)
            assert os.path.isfile(filepath.format(
                epoch=i, batch=j)) != epoch_mode
        callback.on_epoch_end(i, epoch_logs)
        assert os.path.isfile(filepath.format(epoch=i, batch=-1)) == epoch_mode
    callback.on_train_end()
    assert callback.values == [1 for _ in range(10 * 10)]
    assert callback.epoch_values == [10 for _ in range(10)]


@pytest.mark.callback
@pytest.mark.parametrize("filepath, sample_mode, epoch_mode", [
    ('{epoch:02d}/array.npy', 'epoch', True),
    ('{batch:02d}/array.npy', 'batch', False),
])
def test_valuehistory(dirpath, filepath, sample_mode, epoch_mode):
    epoch_logs = {'value': 10}
    batch_logs = {'size': 10, 'value': 1}
    name = 'value'

    filepath = os.path.join(dirpath, filepath)

    callback = cbks.ValueHistory(filepath, name, sample_mode=sample_mode)
    assert callback.filepath == filepath
    assert callback.name == name
    assert callback.epoch_mode == epoch_mode

    callback.on_train_begin()
    assert callback.values == []
    assert callback.epoch_values == []

    for i in range(10):
        callback.on_epoch_begin(i, epoch_logs)
        for j in range(10):
            callback.on_batch_begin(j, batch_logs)
            callback.on_batch_end(j, batch_logs)
            assert os.path.isfile(filepath.format(
                epoch=i, batch=j)) != epoch_mode
        callback.on_epoch_end(i, epoch_logs)
        assert os.path.isfile(filepath.format(epoch=i, batch=-1)) == epoch_mode
    callback.on_train_end()
    assert callback.values == [1 for _ in range(10 * 10)]
    assert callback.epoch_values == [10 for _ in range(10)]

@pytest.mark.callback
@pytest.mark.parametrize('epochs, steps', [
    (1, 10),
    (2, 10),
    (30, 50),
])
def test_progbar(epochs, steps):
    progbar = cbks.GanProgbarLogger()
    progbar.params = {
        'epochs': epochs,
        'steps': steps,
        'metrics': {
            'discriminator': ['loss'],
            'generator': ['loss']
        }
    }
    progbar.on_train_begin()
    for epoch in range(epochs):
        progbar.on_epoch_begin(epoch)
        for step in range(steps):
            logs = {
                'discriminator': {
                    'loss': step,
                },
                'generator': {
                    'loss': step,
                }
            }
            progbar.on_batch_begin(step, logs)
            progbar.on_batch_end(step, logs)
        logs = {
            'discriminator': {
                'loss': epoch,
            },
            'generator': {
                'loss': epoch,
            }
        }
        progbar.on_epoch_end(epoch, logs)

    progbar.on_train_end()

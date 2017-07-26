import tempfile
from unittest import TestCase, skip

import numpy as np
import tensorflow as tf
from ganutil import fit_generator
from keras.callbacks import Callback
from keras.layers import Activation, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence


class TestFitGenerator(TestCase):
    """ganutil.fit_generator()のテストを行うクラス."""

    def setUp(self):
        generator = Sequential()
        generator.add(Dense(16, input_shape=(32,)))
        generator.add(Activation('tanh'))
        generator.add(Reshape((4, 4, 1)))
        self.generator = generator
        self.generator._make_predict_function()
        self.generator_graph = tf.get_default_graph()

        discriminator = Sequential()
        discriminator.add(Flatten(input_shape=(4, 4, 1)))
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))
        self.discriminator = discriminator
        self.discriminator.compile(
            Adam(), 'binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False

        gan = Sequential((self.generator, self.discriminator))
        self.gan = gan
        self.gan.compile(Adam(), 'binary_crossentropy', metrics=['accuracy'])

    def test_fit_generator(self):
        """ganutil.fit_generatorでganが訓練されることを確認する"""
        def d_generator():
            while True:
                with self.generator_graph.as_default():
                    ginputs = np.random.uniform(-1,
                                                1, [5, 32]).astype(np.float32)
                    inputs = self.generator.predict_on_batch(ginputs)
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

        epochs = 5
        steps_per_epoch = 10
        fit_generator(self.gan, self.discriminator, self.generator,
                      d_generator(), g_generator(), steps_per_epoch,
                      epochs=epochs, initial_epoch=0)

    def test_callbacks(self):
        """コールバックが正しい回数呼び出されることを確認する."""
        def d_generator():
            while True:
                with self.generator_graph.as_default():
                    ginputs = np.random.uniform(-1,
                                                1, [5, 32]).astype(np.float32)
                    inputs = self.generator.predict_on_batch(ginputs)
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

        df = tempfile.TemporaryFile(mode='a+')

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
                df.write('D:on_train_begin %d' % this.count_on_train_begin)
                self.assertDictEqual(logs, {})

            def on_train_end(this, logs={}):
                this.count_on_train_end += 1
                df.write('D:on_train_end %d' % this.count_on_train_end)

            def on_epoch_begin(this, epoch, logs={}):
                this.count_on_epoch_begin += 1
                df.write('D:on_epoch_begin %d' % this.count_on_epoch_begin)
                self.assertDictEqual(logs, {})

            def on_epoch_end(this, epoch, logs={}):
                this.count_on_epoch_end += 1
                df.write('D:on_epoch_end %d' % this.count_on_epoch_end)
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

            def on_batch_begin(this, batch, logs={}):
                this.count_on_batch_begin += 1
                df.write('D:on_batch_begin %d' % this.count_on_batch_begin)
                self.assertTrue('size' in logs)

            def on_batch_end(this, batch, logs={}):
                this.count_on_batch_end += 1
                df.write('D:on_batch_end %d' % this.count_on_batch_end)
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

        gf = tempfile.TemporaryFile(mode='a+')

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
                df.write('G:on_train_begin %d' % this.count_on_train_begin)
                self.assertDictEqual(logs, {})

            def on_train_end(this, logs={}):
                this.count_on_train_end += 1
                df.write('G:on_train_end %d' % this.count_on_train_end)

            def on_epoch_begin(this, epoch, logs={}):
                this.count_on_epoch_begin += 1
                df.write('G:on_epoch_begin %d' % this.count_on_epoch_begin)
                self.assertDictEqual(logs, {})

            def on_epoch_end(this, epoch, logs={}):
                this.count_on_epoch_end += 1
                df.write('G:on_epoch_end %d' % this.count_on_epoch_end)
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

            def on_batch_begin(this, batch, logs={}):
                this.count_on_batch_begin += 1
                df.write('G:on_batch_begin %d' % this.count_on_batch_begin)
                self.assertTrue('size' in logs)

            def on_batch_end(this, batch, logs={}):
                this.count_on_batch_end += 1
                df.write('G:on_batch_end %d' % this.count_on_batch_end)
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

        d_callback = DCallback()
        g_callback = GCallback()
        epochs = 5
        steps_per_epoch = 10
        d_iteration_per_step = 3
        g_iteration_per_step = 2
        fit_generator(self.gan, self.discriminator, self.generator,
                      d_generator(), g_generator(), steps_per_epoch,
                      d_iteration_per_step=d_iteration_per_step,
                      g_iteration_per_step=g_iteration_per_step,
                      epochs=epochs, d_callbacks=[d_callback],
                      g_callbacks=[g_callback])
        df.close()
        gf.close()
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
                    with self.generator_graph.as_default():
                        ginputs = np.random.uniform(-1, 1,
                                                    [this.batch_size, 32]).astype(np.float32)
                        inputs = self.generator.predict_on_batch(ginputs)
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
                with self.generator_graph.as_default():
                    ginputs = np.random.uniform(-1,
                                                1, [5, 32]).astype(np.float32)
                    inputs = self.generator.predict_on_batch(ginputs)
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
                self.assertDictEqual(logs, {})

            def on_train_end(this, logs={}):
                this.count_on_train_end += 1

            def on_epoch_begin(this, epoch, logs={}):
                this.count_on_epoch_begin += 1
                self.assertDictEqual(logs, {})

            def on_epoch_end(this, epoch, logs={}):
                this.count_on_epoch_end += 1
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

            def on_batch_begin(this, batch, logs={}):
                this.count_on_batch_begin += 1
                self.assertTrue('size' in logs)

            def on_batch_end(this, batch, logs={}):
                this.count_on_batch_end += 1
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

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
                self.assertDictEqual(logs, {})

            def on_train_end(this, logs={}):
                this.count_on_train_end += 1

            def on_epoch_begin(this, epoch, logs={}):
                this.count_on_epoch_begin += 1
                self.assertDictEqual(logs, {})

            def on_epoch_end(this, epoch, logs={}):
                this.count_on_epoch_end += 1
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

            def on_batch_begin(this, batch, logs={}):
                this.count_on_batch_begin += 1
                self.assertTrue('size' in logs)

            def on_batch_end(this, batch, logs={}):
                this.count_on_batch_end += 1
                self.assertTrue('loss' in logs)
                self.assertTrue('acc' in logs)

        d_callback = DCallback()
        g_callback = GCallback()
        epochs = 5
        steps_per_epoch = 10
        d_iteration_per_step = 3
        g_iteration_per_step = 2
        fit_generator(self.gan, self.discriminator, self.generator,
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
        fit_generator(self.gan, self.discriminator, self.generator,
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

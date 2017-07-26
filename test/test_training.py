import tempfile
from unittest import TestCase

import numpy as np
import tensorflow as tf
from ganutil import fit_generator
from keras.callbacks import Callback
from keras.layers import Activation, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam


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
        pass

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
        fit_generator(self.gan, self.discriminator, self.generator,
                      d_generator(), g_generator(), steps_per_epoch,
                      epochs=epochs, d_callbacks=[d_callback],
                      g_callbacks=[g_callback])
        df.close()
        gf.close()
        self.assertEqual(d_callback.count_on_train_begin, 1)
        self.assertEqual(d_callback.count_on_train_end, 1)
        self.assertEqual(d_callback.count_on_epoch_begin, epochs)
        self.assertEqual(d_callback.count_on_epoch_end, epochs)
        self.assertEqual(d_callback.count_on_batch_begin,
                         steps_per_epoch * epochs)
        self.assertEqual(d_callback.count_on_batch_end,
                         steps_per_epoch * epochs)

        self.assertEqual(g_callback.count_on_train_begin, 1)
        self.assertEqual(g_callback.count_on_train_end, 1)
        self.assertEqual(g_callback.count_on_epoch_begin, epochs)
        self.assertEqual(g_callback.count_on_epoch_end, epochs)
        self.assertEqual(g_callback.count_on_batch_begin,
                         steps_per_epoch * epochs)
        self.assertEqual(g_callback.count_on_batch_end,
                         steps_per_epoch * epochs)

    def test_steps_per_epoch(self):
        """1エポック中に指定された回数訓練されることを確認する."""
        pass

    def test_iteration_per_step(self):
        """1ステップ中に指定された回数訓練されることを確認する."""
        pass

    def test_generator(self):
        """データのジェネレーターが動作することを確認する."""
        pass

    def test_multi_threading(self):
        """データのジェネレートを並列で動作できることを確認"""
        pass

    def test_multi_processing(self):
        """データのジェネレートをマルチプロセスで動作できることを確認"""

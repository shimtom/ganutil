# -*- coding: utf-8 -*-
import warnings
from math import ceil

import keras.callbacks as cbks
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import GeneratorEnqueuer, OrderedEnqueuer, Sequence

from .callbacks import ProgbarLogger as GanProgbarLogger
from .saver import Saver


default_preprocessor = ImageDataGenerator()
default_saver = Saver('save')


def fit_generator(gan, discriminator, generator, d_generator, g_generator,
                  steps_per_epoch, d_iteration_per_step=1, g_iteration_per_step=1,
                  epochs=1, d_callbacks=None, g_callbacks=None,
                  max_queue_size=10, workers=1, use_multiprocessing=False,
                  shuffle=True, initial_epoch=0):

    d_is_sequence = isinstance(d_generator, Sequence)
    g_is_sequence = isinstance(g_generator, Sequence)

    if not (d_is_sequence and g_is_sequence) and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    d_enqueuer = None
    g_enqueuer = None

    wait_time = 0.01  # in seconds

    try:
        if d_is_sequence:
            d_enqueuer = OrderedEnqueuer(
                d_generator, use_multiprocessing=use_multiprocessing, shuffle=shuffle)
        else:
            d_enqueuer = GeneratorEnqueuer(
                d_generator, use_multiprocessing=use_multiprocessing, wait_time=wait_time)
        d_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        d_sample_generator = d_enqueuer.get()

        if g_is_sequence:
            g_enqueuer = OrderedEnqueuer(
                g_generator, use_multiprocessing=use_multiprocessing, shuffle=shuffle)
        else:
            g_enqueuer = GeneratorEnqueuer(
                g_generator, use_multiprocessing=use_multiprocessing, wait_time=wait_time)
        g_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        g_sample_generator = g_enqueuer.get()

        discriminator.history = cbks.History()
        # BaseLoggerは1番目でなければならない
        d_callbacks = [cbks.BaseLogger()] + (d_callbacks or [])
        d_callbacks += [discriminator.history]
        d_callbacks += [GanProgbarLogger(name='Discriminator',
                                         count_mode='steps')]
        for c in d_callbacks:
            if isinstance(c, cbks.ProgbarLogger):
                warnings.warn(UserWarning('Using a `keras.callbacks.ProgbarLogger, `'
                                          ' it can\'t distinguishe whether output is generator\'s or discriminator\'s.'
                                          ' Please consider using the`ganutil.callbacks.ProgbarLogger'
                                          ' class.'))

        d_callbacks = cbks.CallbackList(d_callbacks)
        d_callbacks.set_model(discriminator)
        d_callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1,
            'do_validation': False,
            'metrics': discriminator.metrics_names,
        })

        gan.history = cbks.History()
        # BaseLoggerは1番目でなければならない
        g_callbacks = [cbks.BaseLogger()] + (g_callbacks or []) + [gan.history]
        g_callbacks += [GanProgbarLogger(name='Generator', count_mode='steps')]
        for c in g_callbacks:
            if isinstance(c, cbks.ProgbarLogger):
                warnings.warn(UserWarning('Using a `keras.callbacks.ProgbarLogger, `'
                                          ' it can\'t distinguishe whether output is generator\'s or discriminator\'s.'
                                          ' Please consider using the`ganutil.callbacks.ProgbarLogger'
                                          ' class.'))
            if isinstance(c, cbks.ModelCheckpoint):
                warnings.warn(UserWarning('Using a `keras.callbacks.ModelCheckpoint, `'
                                          ' it can\'t save only generator model.'
                                          ' Please consider using the`ganutil.callbacks.GanModelCheckpoint'
                                          ' class.'))

        g_callbacks = cbks.CallbackList(g_callbacks)
        g_callbacks.set_model(gan)
        g_callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1,
            'do_validation': False,
            'metrics': gan.metrics_names,
        })

        d_callbacks.on_train_begin()
        g_callbacks.on_train_begin()

        d_epoch_logs = {}
        g_epoch_logs = {}
        for epoch in range(initial_epoch, epochs):
            d_callbacks.on_epoch_begin(epoch, d_epoch_logs)
            g_callbacks.on_epoch_begin(epoch, g_epoch_logs)
            for step in range(steps_per_epoch):
                d_batch_logs = {}
                for index in range(d_iteration_per_step):
                    samples = next(d_sample_generator)
                    d_batch_logs['batch'] = step
                    d_batch_logs['iteration'] = index
                    d_batch_logs['size'] = samples[0].shape[0]
                    d_callbacks.on_batch_begin(step, d_batch_logs)
                    d_outs = discriminator.train_on_batch(*samples)
                    if not isinstance(d_outs, list):
                        d_outs = [d_outs]
                    for n, o in zip(discriminator.metrics_names, d_outs):
                        d_batch_logs[n] = o
                    d_callbacks.on_batch_end(step, d_batch_logs)

                g_batch_logs = {}
                for index in range(g_iteration_per_step):
                    samples = next(g_sample_generator)
                    g_batch_logs['batch'] = step
                    g_batch_logs['iteration'] = index
                    g_batch_logs['size'] = samples[0].shape[0]
                    g_callbacks.on_batch_begin(step, g_batch_logs)
                    g_outs = gan.train_on_batch(*samples)
                    if not isinstance(g_outs, list):
                        g_outs = [g_outs]
                    for n, o in zip(gan.metrics_names, d_outs):
                        g_batch_logs[n] = o
                    g_callbacks.on_batch_end(step, g_batch_logs)

                d_callbacks.on_epoch_end(epoch, d_epoch_logs)
                g_callbacks.on_epoch_end(epoch, g_epoch_logs)

            d_callbacks.on_train_end()
            g_callbacks.on_train_end()

    finally:
        if d_enqueuer is not None:
            d_enqueuer.stop()
        if g_enqueuer is not None:
            g_enqueuer.stop()

    return discriminator.history, gan.history


def train(gan, discriminator, generator, d_inputs, g_inputs, epoch_size, batch_size=32, d_callbacks=None, g_callbacks=None, preprocessor=default_preprocessor):
    """GANを訓練する.
    また,エポックごとに学習結果を保存する.それぞれの損失,精度のグラフ,モデル,パラメータ,生成画像が保存される.保存にはgan.saverを使用する.
    :param keras.Model gan: compile済みgenerator + discriminatorモデル.
        generatorは訓練可能でなければならないがdiscriminatorは訓練可能であってはならない.
    :param keras.Model discriminator: compile済みdiscriminatorモデル. 訓練可能でなければならない.
        出力の形状は(data size, 1)で値は[0, 1]の範囲でなければならない.
    :param keras.Model generator: ganに使用したgeneratorモデル.
        出力の形状は(size, height, width, ch)で各値は[-1, 1]の範囲でなければならない.
    :param numpy.ndarray d_inputs: discriminatorの学習に使用する入力データセット.
    :param numpy.ndarray g_inputs: discriminatorの学習に使用する入力データセット.
    :param int epoch_size: 最大のエポック数.
    :param int batch_size: バッチの大きさ.デフォルトは`32`.
    :param keras.preprocessing.image.ImageDataGenerator preprocessor:
        discriminatorの入力データに対して前処理を行ったデータのジェネレーター.
        デフォルトは何もしないジェネレーターを設定している.
    :param Saver saver: 各値を保存するセーバー.
        デフォルトは`save`ディレクトリに各値を保存する.
    """
    step_size = int(ceil(len(d_inputs) / batch_size))

    discriminator.history = cbks.History()
    # BaseLoggerは1番目でなければならない
    d_callbacks = [cbks.BaseLogger()] + (d_callbacks or [])
    d_callbacks += [discriminator.history]
    d_callbacks += [GanProgbarLogger(name='Discriminator', count_mode='steps')]
    for c in d_callbacks:
        if isinstance(c, cbks.ProgbarLogger):
            warnings.warn(UserWarning('Using a `keras.callbacks.ProgbarLogger, `'
                                      ' it can\'t distinguishe whether output is generator\'s or discriminator\'s.'
                                      ' Please consider using the`ganutil.callbacks.ProgbarLogger'
                                      ' class.'))
    d_callbacks = cbks.CallbackList(d_callbacks)
    d_callbacks.set_model(discriminator)
    d_callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epoch_size,
        'samples': batch_size,
        'verbose': 1,
        'do_validation': False,
        'metrics': discriminator.metrics_names,
    })

    gan.history = cbks.History()
    # BaseLoggerは1番目でなければならない
    g_callbacks = [cbks.BaseLogger()] + (g_callbacks or [])
    g_callbacks += [gan.history]
    g_callbacks += [GanProgbarLogger(name='Generator', count_mode='steps')]
    for c in g_callbacks:
        if isinstance(c, cbks.ProgbarLogger):
            warnings.warn(UserWarning('Using a `keras.callbacks.ProgbarLogger, `'
                                      ' it can\'t distinguishe whether output is generator\'s or discriminator\'s.'
                                      ' Please consider using the`ganutil.callbacks.ProgbarLogger'
                                      ' class.'))
        if isinstance(c, cbks.ModelCheckpoint):
            warnings.warn(UserWarning('Using a `keras.callbacks.ModelCheckpoint, `'
                                      ' it can\'t save only generator model.'
                                      ' Please consider using the`ganutil.callbacks.GanModelCheckpoint'
                                      ' class.'))

    g_callbacks = cbks.CallbackList(g_callbacks)
    g_callbacks.set_model(gan)
    g_callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epoch_size,
        'samples': batch_size,
        'verbose': 1,
        'do_validation': False,
        'metrics': gan.metrics_names,
    })

    d_callbacks.on_train_begin()
    g_callbacks.on_train_begin()

    for epoch in range(epoch_size):
        d_callbacks.on_epoch_begin(epoch)
        g_callbacks.on_epoch_begin(epoch)
        d = preprocessor.flow(d_inputs, np.ones(
            len(d_inputs), dtype=np.int64), batch_size=batch_size)
        g = _set_input_generator(g_inputs)
        # discriminator の 入力データジェネレーターを回す
        for step, samples in enumerate(d):
            if step + 1 > step_size:
                break
            # real batch size
            n = len(samples[0])

            # train discriminator
            x = np.concatenate(
                (samples[0], _generate(generator, g(len(samples[0])))))
            y = np.concatenate(
                (samples[1], np.zeros(len(samples[0])))).astype(np.int64)
            d_batch_logs = {'batch': step, 'size': x.shape[0]}
            d_callbacks.on_batch_begin(step, d_batch_logs)
            d_outs = discriminator.train_on_batch(x, y)
            if not isinstance(d_outs, list):
                d_outs = [d_outs]
            for n, o in zip(discriminator.metrics_names, d_outs):
                d_batch_logs[n] = o
            d_callbacks.on_batch_end(step, d_batch_logs)

            # train generator
            x = g(n)
            y = np.ones(n, dtype=np.int64)
            g_batch_logs = {'batch': step, 'size': x.shape[0]}
            g_callbacks.on_batch_begin(step, g_batch_logs)
            g_outs = gan.train_on_batch(x, y)
            if not isinstance(g_outs, list):
                g_outs = [g_outs]
            for n, o in zip(gan.metrics_names, g_outs):
                g_batch_logs[n] = o
            g_callbacks.on_batch_end(step, g_batch_logs)

        d_callbacks.on_epoch_end(epoch)
        g_callbacks.on_epoch_end(epoch)

    d_callbacks.on_train_end()
    g_callbacks.on_train_end()

    return discriminator.history, gan.history


def _set_input_generator(data):
    data_size = len(data)
    indices = np.random.permutation(data_size)
    index = 0

    def generate(size):
        nonlocal index
        s, e = index, index + size
        inputs = data[indices[s:e]]
        index = (index + size) % data_size
        if e > data_size:
            inputs = np.concatenate((inputs, data[:index]))
        return inputs

    return generate


def _generate(generator, inputs, to_image=False):
    generated = np.array(generator.predict_on_batch(inputs))
    if to_image:
        generated = generated * 127.5 + 127.5
        generated = generated.astype(np.uint8)
    return generated

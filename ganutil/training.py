# -*- coding: utf-8 -*-
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from collections import namedtuple
from .saver import Saver
import numpy as np
import time
import copy
import keras.callbacks as cbks

DataSet = namedtuple('DataSet', ['x', 'y', 'size'])
Gan = namedtuple('Gan', ['d', 'g'])

default_preprocessor = ImageDataGenerator()
default_saver = Saver('save')

def train(discriminator, generator, gan, d_inputs, g_inputs, epoch_size, batch_size=32, callbacks=None,
          preprocessor=default_preprocessor, saver=default_saver):
    """GANを訓練する.
    また,エポックごとに学習結果を保存する.それぞれの損失,精度のグラフ,モデル,パラメータ,生成画像が保存される.保存にはgan.saverを使用する.
    :param keras.Model discriminator: compile済みdiscriminatorモデル. 訓練可能でなければならない.
        出力の形状は(data size, 1)で値は[0, 1]の範囲でなければならない.
    :param keras.Model generator: ganに使用したgeneratorモデル.
        出力の形状は(size, height, width, ch)で各値は[-1, 1]の範囲でなければならない.
    :param keras.Model gan: compile済みgenerator + discriminatorモデル.
        generatorは訓練可能でなければならないがdiscriminatorは訓練可能であってはならない.
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
    saver.architecture(discriminator, generator)

    data_size = len(d_inputs)
    step_size = int(ceil(len(d_inputs) / batch_size))

    g_total = dict()
    d_total = dict()

    start = time.time()

    d_out_labels = copy.copy(_get_deduped_metrics_names(discriminator))
    d_history = cbks.History()
    d_callbacks = [cbks.BaseLogger()] + (callbacks or []) + [d_history] + [cbks.ProgbarLogger()]
    d_callbacks = cbks.CallbackList(d_callbacks)
    d_out_labels = d_out_labels or []
    d_callbacks.set_model(discriminator)
    d_callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epoch_size,
            'samples': batch_size,
            'verbose': 1,
            'do_validation': False,
            'metrics': d_out_labels or [],
        })

    g_history = cbks.History()
    g_out_labels = copy.copy(_get_deduped_metrics_names(gan))

    # TODO gan_on_train_begion
    # TODO discriminator_on_train_begion
    # TODO generator_on_train_begion
    d_callbacks.on_train_begin()

    for epoch in range(epoch_size):
        # TODO gan_on_epoch_begion
        # TODO discriminator_on_epoch_begin
        # TODO generator_on_epoch_begion
        d_callbacks.on_epoch_begin(epoch)
        start_epoch = time.time()
        d = preprocessor.flow(d_inputs, np.ones(len(d_inputs), dtype=np.int64), batch_size=batch_size)
        g = _set_input_generator(g_inputs)
        # discriminator の 入力データジェネレーターを回す
        for step, samples in enumerate(d):
            if step + 1 > step_size:
                break
            # real batch size
            n = len(samples[0])

            # FIXME add gan_on_batch_begin()
            batch_logs = {}
            batch_logs['batch'] = step
            batch_logs['size'] = len(samples[0])
            callbacks.on_batch_begin(step, batch_logs)

            start_step = time.time()


            # train discriminator
            # TODO add discriminator_on_batch_begin()
            x = np.concatenate((samples[0], _generate(generator, g(n))))
            y = np.concatenate((samples[1], np.zeros(n))).astype(np.int64)
            d_scalars = discriminator.train_on_batch(x, y)
            if len(discriminator.metrics_names) == 1:
                d_scalars = [d_scalars]
            d_result = dict()
            for s, n in zip(d_scalars, discriminator.metrics_names):
                d_result[n] = s
            # TODO add discriminator_on_batch_end()

            # train generator
            # TODO add generator_on_batch_begion()
            x = g(n)
            y = np.ones(n, dtype=np.int64)
            g_scalars = gan.train_on_batch(x, y)
            if len(gan.metrics_names) == 1:
                g_scalars = [d_scalars]
            g_result = dict()
            for s, n in zip(g_scalars, gan.metrics_names):
                g_result[n] = s
            # TODO add generator_on_batch_end()

            # TODO add gan_on_batch_end()
            # print result per step
            print('time %.3fs epoch %d / %d [%d] d: %s g: %s' % (
                  time.time() - start_step, epoch, epoch_size, str(d_result), str(g_result)))

            # compute sum
            for k, v in d_result.items():
                g_result[k] = g_result.get(k, 0.) + v * n
            for k, v in g_result.items():
                g_result[k] = g_result.get(k, 0.) + v * n
        # TODO add discriminator_on_epoch_end
        # TODO add generator_on_epoch_end
        # TODO add gan_on_epoch_end
        # compute total average
        for k, v in g_result.items():
            d_total.setdefault(k, []).append(g_result.get(k, 0.) / data_size)
        for k, v in g_total.items():
            g_total.setdefault(k, []).append(g_result.get(k, 0.) / data_size)


        # print result
        print('time %.3fs epoch %d / %d d: %s g: %s' % (time.time() - start_epoch, epoch, epoch_size, d_total, g_total))

        # save
        saver.model(discriminator, generator)
        saver.parameter(discriminator, generator)
        saver.scalars(d_total, g_total)
        saver.image(_generate(generator, g(25)), id=epoch)

    # TODO add discriminator_on_train_end
    # TODO add generator_on_train_end
    # TODO add gan_on_train_end
    print('time %.3fs' % (time.time() - start))
    return d_history, g_history

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

def _get_deduped_metrics_names(model):
    out_labels = model.metrics_names

    # Rename duplicated metrics name
    # (can happen with an output layer shared among multiple dataflows).
    deduped_out_labels = []
    for i, label in enumerate(out_labels):
        new_label = label
        if out_labels.count(label) > 1:
            dup_idx = out_labels[:i].count(label)
            new_label += '_' + str(dup_idx + 1)
        deduped_out_labels.append(new_label)
    return deduped_out_labels

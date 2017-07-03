# -*- coding: utf-8 -*-
from math import ceil
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from collections import namedtuple
from .saver import Saver
import numpy as np

DataSet = namedtuple('DataSet', ['x', 'y', 'size'])
Gan = namedtuple('Gan', ['d', 'g'])

default_preprocessor = ImageDataGenerator()
default_saver = Saver('save')

def train(discriminator, generator, d_opt, g_opt, d_inputs, g_inputs, epoch_size, batch_size=32,
          preprocessor=default_preprocessor, saver=default_saver):
    """GANを訓練する.
    また,エポックごとに学習結果を保存する.それぞれの損失,精度のグラフ,モデル,パラメータ,生成画像が保存される.保存にはgan.saverを使用する.

    :param keras.Model discriminator: discriminatorモデル.
        出力の形状は(data size, 1)で値は[0, 1]の範囲でなければならない.
    :param keras.Model generator: generatorモデル.
        出力の形状は(size, height, width, ch)で各値は[-1, 1]の範囲でなければならない.
    :param keras.Optimizer d_opt: discriminatorの学習に使用する最適化.
    :param keras.Optimizer g_opt: generatorの学習に使用する最適化.
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
    # compile for discriminator training
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=d_opt,
                          metrics=['accuracy'])

    # compile for generator training
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy',
                optimizer=g_opt,
                metrics=['accuracy'])

    # save model architecture
    saver.architecture(discriminator, generator)

    losses = Gan([], [])
    accuracies = Gan([], [])
    data_size = len(d_inputs)
    step_size = int(ceil(len(d_inputs) / batch_size))

    for epoch in range(epoch_size):
        d_loss, g_loss = 0., 0.
        d_acc, g_acc = 0., 0.
        d = preprocessor.flow(d_inputs, np.ones(len(d_inputs), dtype=np.int64), batch_size=batch_size)
        g = set_input_generator(g_inputs)
        # discriminator の 入力データジェネレーターを回す
        for step, samples in enumerate(d):
            if step + 1 > step_size:
                break

            # real batch size
            n = len(samples[0])

            # train discriminator
            x = np.concatenate((samples[0], generate(generator, g(n))))
            y = np.concatenate((samples[1], np.zeros(n))).astype(np.int64)
            dl, da = discriminator.train_on_batch(x, y)

            # train generator
            x = g(n)
            y = np.ones(n, dtype=np.int64)
            gl, ga = gan.train_on_batch(x, y)

            # print result per step
            print('epoch %d / %d [%d] loss(d, g): (%f, %f) acc(d, g): (%.3f, %.3f)' % (epoch, epoch_size, step, dl, gl, da, ga))

            # compute toal loss and accuracy
            d_loss, g_loss = d_loss + dl * n, g_loss + gl * n
            d_acc, g_acc = d_acc + da * n, g_acc + ga * n

        d_loss /= data_size
        g_loss /= data_size
        losses.d.append(d_loss)
        losses.g.append(g_loss)

        g_acc /= data_size
        d_acc /= data_size
        accuracies.g.append(g_acc)
        accuracies.d.append(d_acc)

        # print result
        print('epoch %d / %d loss(d, g): (%f, %f) acc(d, g): (%.3f, %.3f)' % (epoch, epoch_size, d_loss, g_loss, d_acc, g_acc))

        # save
        saver.model(discriminator, generator)
        saver.parameter(discriminator, generator)
        saver.loss(*losses)
        saver.accuracy(*accuracies)
        saver.image(generate(generator, g(25)), id=epoch)

def set_input_generator(data):
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

def generate(generator, inputs, to_image=False):
    generated = np.array(generator.predict_on_batch(inputs))
    if to_image:
        generated = generated * 127.5 + 127.5
        generated = generated.astype(np.uint8)
    return generated

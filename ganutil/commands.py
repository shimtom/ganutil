# -*- coding: utf-8 -*-
from keras.models import Sequential, model_from_config, load_model
from keras.optimizers import Adam
from json import load as json_load
from yaml import load as yaml_load
from sys import exit, stderr
from numpy import array_equal, load

from .saver import Saver
from .training import train as train_gan
from .generation import generate as gan_generate
from .discrimination import discriminate as gan_discriminate
from .ensure_existing import ensure_file, ensure_files, ensure_directory, ensure_input_shape

def train(args):
    discriminator_path = args.discriminator
    discriminator_input_path = args.dinput
    discriminator_weight_path = args.dweight
    discriminator_learning_rate = args.dlr
    discriminator_beta_1 = args.dbeta1

    generator_path = args.generator
    generator_input_path = args.ginputs
    generator_weight_path = args.gweight
    generator_learning_rate = args.glr
    generator_beta_1 = args.gbeta1

    save_path = args.save
    epoch_size = args.epoch
    batch_size = args.batch

    # 引数で指定されたパスが存在することを確かめる
    if not (ensure_files([discriminator_path, generator_path], extensions=['json', 'yml']) and
            ensure_files([discriminator_input_path, generator_input_path], extensions=['npy']) and
            ensure_directory(save_path)):
        exit(1)

    # discriminatorモデルと入力データを読み込み、形状が正しいかを確かめる.
    discriminator = _load_model_architecture_and_weight(discriminator_path, discriminator_weight_path)
    discriminator_inputs = load(discriminator_input_path)
    if not ensure_input_shape(discriminator.inputs[0].shape[1:], discriminator_inputs.shape[1:], 'discriminator'):
        exit(1)
    # discriminatorの出力層の形状を確かめる.
    discriminator_output_shape = discriminator.outputs[-1].shape[1:]
    if not (len(discriminator_output_shape) == 1 and discriminator_output_shape[0] == 1):
        print('discriminator output shape %s is not supported' % (str(discriminator_output_shape)), file=stderr)
        exit(1)

    # generatorモデルと入力データを読み込み、形状が正しいかを確かめる.
    generator = _load_model_architecture_and_weight(generator_path, generator_weight_path)
    generator_inputs = load(generator_input_path)
    if not ensure_input_shape(generator.inputs[0].shape[1:], generator_inputs.shape[1:], 'generator'):
        exit(1)
    # generatorの出力層の形状を確かめる
    generator_output_shape = generator.outputs[-1].shape[1:]
    if not array_equal(generator_output_shape, discriminator_inputs.shape[1:]):
        print('generator output shape %s is not valid' % (str(generator_output_shape)), file=stderr)
        exit(1)

    saver = Saver(save_path)

    d_opt = Adam(lr=discriminator_learning_rate, beta_1=discriminator_beta_1)
    g_opt = Adam(lr=generator_learning_rate, beta_1=generator_beta_1)

    # 学習に使用した各種パラメータを保存する.
    config = {
        'dataset': {
            'discriminator': discriminator_input_path,
            'generator': generator_input_path
        },
        'epochSize': epoch_size,
        'batchSize': batch_size,
        'saveDirectory': save_path,
        'discriminator': {
            'architecture': discriminator_path,
            'weight': discriminator_weight_path,
            'adam': {
                'lr': discriminator_learning_rate,
                'beta_1': discriminator_beta_1,
                'beta_2': 0.999,
                'epsilon': 1e-08,
                'decay': 0.
            }
        },
        'generator': {
            'architecture': generator_path,
            'weight': generator_weight_path,
            'adam': {
                'lr': generator_learning_rate,
                'beta_1': generator_learning_rate,
                'beta_2': 0.999,
                'epsilon': 1e-08,
                'decay': 0.
            }
        }
    }
    saver.config(config)

    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

    # compile for generator training
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=g_opt, metrics=['accuracy'])
    discriminator.trainable = True

    # 訓練開始
    train_gan(discriminator, generator, gan, discriminator_inputs, generator_inputs,
              epoch_size, batch_size=batch_size, saver=saver)

def generate(args):
    generator_path = args.model
    dataset_path = args.x
    save_path = args.save
    batch_size = args.batch

    if not ensure_file(generator_path, extensions=['h5']):
        exit(1)
    if not ensure_file(dataset_path, extensions=['npy']):
        exit(1)
    if not ensure_directory(save_path):
        exit(1)

    generator = load_model(generator_path)
    dataset = load(dataset_path)

    if not ensure_input_shape(generator.inputs[0].shape[1:], dataset.shape[1:], 'generator'):
        exit(1)

    gan_generate(generator, dataset, save_path, batch_size=batch_size)


def discriminate(args):
    discriminator_path = args.model
    dataset_path = args.x
    save_path = args.save
    batch_size = args.batch

    if not ensure_file(discriminator_path, extensions=['h5']):
        exit(1)
    if not ensure_file(dataset_path, extensions=['npy']):
        exit(1)
    if not ensure_directory(save_path):
        exit(1)

    discriminator = load_model(discriminator_path)
    dataset = load(dataset_path)
    if not ensure_input_shape(discriminator.inputs[0].shape[1:], dataset.shape[1:], 'discriminator'):
        exit(1)

    gan_discriminate(discriminator, dataset, save_path, batch_size=batch_size)

def _load_model_architecture_and_weight(architecture_path, weight_path):
    """モデルのアーキテクチャと重みパラメータを読み込む.

    :param str architecture_path: モデルのアーキテクチャを保存したファイルのパス.
    :param str weight_path: モデルの重みパラメータを保存したファイルのパス.
                            Noneを指定した場合は重みパラメータを読み込まない.
    :return keras.Model: モデル.
    """
    model = _load_model_architecture(architecture_path)
    if weight_path is not None:
        if not ensure_file(weight_path, extensions=['h5']):
            exit(1)
        model.load_weights(weight_path)

    return model

def _load_model_architecture(path):
    """ファイルからモデルのアーキテクチャを読み込む.
    読み込めるファイルは拡張子がjsonかymlであり、ファイルの書式がkerasの仕様に従っている必要がある。

    :param str path: モデルのアーキテクチャを保存したファイルのパス
    :return keras.Model model: モデル.
    """
    architecture = None
    extension = path.split('.')[-1]
    if extension not in ['json', 'yml']:
        raise ValueError('ファイルの拡張子はjsonまたはymlでなければなりません. %s' % path)

    if extension == 'json':
        with open(path, 'r+') as f:
            architecture = json_load(f)
    if extension == 'yml':
        with open(path, 'r+') as f:
            architecture = yaml_load(f)
    model = model_from_config(architecture)

    return model

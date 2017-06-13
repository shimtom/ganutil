import importlib

def load_model_architecture(path):
    """ファイルからモデルのアーキテクチャを読み込む.
    読み込めるファイルは拡張子がjsonかymlであり、ファイルの書式がkerasの仕様に従っている必要がある。

    :param str path: モデルのアーキテクチャを保存したファイルのパス
    :return keras.Model model: モデル.
    """
    from keras.models import model_from_config
    from json import load as json_load
    from yaml import load as yaml_load

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

def ensure_files(paths, extensions=[]):
    """ファイルの存在を確認する.

    :param list of str paths: ファイルのパスのリスト.
    :param list of str extensions: 全てのファイルが持つべき拡張子を指定する.
    :return bool: 全てのファイルが確認できた場合のみTrue.それ以外はFalse.
    """
    result = True
    for path in paths:
        result = result and ensure_file(path, extensions=extensions)
    return result

def ensure_file(path, extensions=[]):
    """ファイルの存在を確認する.

    :param str paths: ファイルのパス.
    :param str extensions: ファイルが持つべき拡張子を指定する.
    :return bool: ファイルが確認できた場合のみTrue.それ以外はFalse.
    """
    from os.path import isfile
    from sys import stderr

    if not isfile(path):
        print('%s is not valid.' % path, file=stderr)
        return False
    if len(extensions) > 0 and path.split('.')[-1] not in extensions:
        print('%s is not supported' % path)
        return False
    return True

def ensure_directory(path):
    """ディレクトリの存在を確認する.

    :param str path: ディレクトリのパス.
    :return bool: ディレクトリが確認できたらTrue.そうでなければFalse.
    """
    from os.path import isdir
    from sys import stderr

    if not isdir(path):
        print('%s is not valid.' % path, file=stderr)
        return False
    return True

def load_model_architecture_and_weight(architecture_path, weight_path):
    """モデルのアーキテクチャと重みパラメータを読み込む.

    :param str architecture_path: モデルのアーキテクチャを保存したファイルのパス.
    :param str weight_path: モデルの重みパラメータを保存したファイルのパス.
                            Noneを指定した場合は重みパラメータを読み込まない.
    :return keras.Model: モデル.
    """
    from sys import exit

    model = load_model_architecture(architecture_path)
    if weight_path is not None:
        if not ensure_file(weight_path, extensions=['h5']):
            exit(1)
        model.load_weights(weight_path)

    return model

def ensure_input_shape(input_shape, data_shape, name):
    """モデルの入力層の形状とデータの形状が等しいことを確かめる.
    :param input_shape: モデルの入力層の形状.
    :param data_shape: データの形状.
    :param name: モデル名.
    :return: モデルの入力層の形状とデータの形状が等しいかどうか.
    """
    from numpy import array_equal
    from sys import stderr

    if not array_equal(input_shape, data_shape):
        print('input data shape %s does not equal to %s input layer shape %s' %
              (str(data_shape), name, str(input_shape)), file=stderr)
        return False
    return True


def command_train(args):
    from keras.optimizers import Adam
    from numpy import load, array_equal
    from .training import train
    from .saver import Saver
    from sys import exit, stderr


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
            ensure_files([discriminator_input_path, generator_input_path], extensions=['npy', 'npz']) and
            ensure_directory(save_path)):
        exit(1)

    # discriminatorモデルと入力データを読み込み、形状が正しいかを確かめる.
    discriminator = load_model_architecture_and_weight(discriminator_path, discriminator_weight_path)
    discriminator_inputs = load(discriminator_input_path)
    if not ensure_input_shape(discriminator.inputs[0].shape[1:], discriminator_inputs.shape[1:], 'discriminator'):
        exit(1)
    # discriminatorの出力層の形状を確かめる.
    discriminator_output_shape = discriminator.outputs[-1].shape[1:]
    if not (len(discriminator_output_shape) == 1 and discriminator_output_shape[0] == 1):
        print('discriminator output shape %s is not supported' % (str(discriminator_output_shape)), file=stderr)
        exit(1)

    # generatorモデルと入力データを読み込み、形状が正しいかを確かめる.
    generator = load_model_architecture_and_weight(generator_path, generator_weight_path)
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

    # 訓練開始
    train(discriminator, generator, d_opt, g_opt, discriminator_inputs, generator_inputs,
          epoch_size, batch_size=batch_size, saver=saver)

def command_generate(args):
    from keras.models import load_model
    from .generation import generate
    from .saver import Saver
    from numpy import load, array_equal
    from sys import exit, stderr

    generator_path = args.model
    dataset_path = args.x
    save_path = args.save
    batch_size = args.batch

    if not ensure_file(generator_path, extensions=['h5']):
        exit(1)

    if not ensure_file(dataset_path, extensions=['npy', 'npz']):
        exit(1)

    if not ensure_directory(save_path):
        exit(1)

    generator = load_model(generator_path)
    dataset = load(dataset_path)
    data_shape = dataset.shape[1:]
    input_shape = generator.inputs[0].shape[1:]
    if not array_equal(data_shape, input_shape):
        print('input data shape %s does not equal to generator input layer shape %s' %
              (str(data_shape), str(input_shape)), file=stderr)
        exit(1)
    saver = Saver(dataset_path, name='generative')

    generate(generator, dataset, batch_size=batch_size, saver=saver)


def command_discriminate(args):
    from keras.models import load_model
    from .discrimination import discriminate
    from .saver import Saver
    from numpy import load, array_equal
    from sys import exit, stderr

    discriminator_path = args.model
    dataset_path = args.x
    save_path = args.save
    batch_size = args.batch

    if not ensure_file(discriminator_path, extensions=['h5']):
        exit(1)
    if not ensure_file(dataset_path, extensions=['npy', 'npz']):
        exit(1)
    if not ensure_directory(save_path):
        exit(1)

    discriminator = load_model(discriminator_path)
    dataset = load(dataset_path)
    data_shape = dataset.shape[1:]
    input_shape = discriminator.inputs[0].shape[1:]
    if not array_equal(data_shape, input_shape):
        print('input data shape %s does not equal to discriminate input layer shape %s' %
              (str(data_shape), str(input_shape)), file=stderr)
        exit(1)

    saver = Saver(save_path, name='generative')

    discriminate(discriminator, dataset, batch_size=batch_size, saver=saver)


def parse_arg():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Generative Adversarial Nets.')
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='GANを訓練する.')
    train_parser.add_argument('discriminator', type=str, help='discriminatorモデル.'
                                                              'サポートしているファイルフォーマットは[.json|.yml].'
                                                              'kerasの仕様に従ったものでなければならない.')
    train_parser.add_argument('generator', type=str, help='generatorモデル.'
                                                          'サポートしているファイルフォーマットは[.json|.yml].'
                                                          'kerasの仕様に従ったものでなければならない.')
    train_parser.add_argument('dinput', type=str, help='discriminatorの訓練に使用する入力データセットのファイル名.'
                                                       'サポートしているファイルフォーマットは[.npy|.npz].')
    train_parser.add_argument('ginput', type=str, help='generatorの訓練に使用する入力データセットのファイル名.'
                                                       'サポートしているファイルフォーマットは[.npy|.npz].')
    train_parser.add_argument('save', type=str, help='学習結果を保存するディレクトリ.'
                                                     '存在しない場合は終了する.')
    train_parser.add_argument('--epoch', default=20,type=int, help='エポックサイズ.デフォルトは20回.')
    train_parser.add_argument('--batch', default=32, type=int, help='バッチサイズ.デフォルトは32.')
    train_parser.add_argument('--dweight', type=str, help='discriminatorの学習済み重みパラメータ.'
                                                          '指定したdiscriminatorのアーキテクチャでなければならない.'
                                                          'サポートしているファイルフォーマットは[.h5].')
    train_parser.add_argument('--dlr', default=0.0002, type=float, help='discriminatorの学習係数.')
    train_parser.add_argument('--dbeta1', default=0.5, type=float, help='discriminatorのAdam Optimizerのbeta1の値.'
                                                                        'デフォルトは0.5.')
    train_parser.add_argument('--gweight', type=str, help='generatorの学習済み重みパラメータ.'
                                                          '指定したgeneratorのアーキテクチャでなければならない.'
                                                          'サポートしているファイルフォーマットは[.h5].')
    train_parser.add_argument('--glr', default=0.0002, type=float, help='generatorの学習係数.デフォルトは0.0002.')
    train_parser.add_argument('--gbeta1', default=0.5, type=float, help='generatorのAdam Optimizerのbeta1の値.'
                                                                        'デフォルトは0.5.')
    train_parser.set_defaults(func=command_train)

    discriminative_parser = subparsers.add_parser('discriminate', help='学習済みのDiscriminatorモデルを用いて識別を行う.')
    discriminative_parser.add_argument('model', type=str, help='学習済みdiscriminatorモデル.'
                                                               'サポートしているファイルフォーマットは[.h5].'
                                                               'kerasの仕様に従ったものでなければならない.')
    discriminative_parser.add_argument('x', type=str, help='識別に使用されるデータセット.'
                                                           'サポートしているファイルフォーマットは[.npy|.npz].')
    discriminative_parser.add_argument('save', type=str, help='結果を保存するディレクトリ.存在しない場合は終了する.'
                                                              'ファイル名は`discriminated.npz`となる.'
                                                              'すでに存在する場合はファイル名に索引を付ける.')
    discriminative_parser.add_argument('-b', '--batch', default=32, type=int, help='バッチサイズ.デフォルトは32.')
    discriminative_parser.set_defaults(func=command_discriminate)

    generative_parser = subparsers.add_parser('generate', help='学習済みのGeneratorモデルを用いて生成を行う.')
    generative_parser.add_argument('model', type=str, help='学習済みgeneratorモデル.'
                                                           'サポートしているファイルフォーマットは[.h5].'
                                                           'kerasの仕様に従ったものでなければならない.')
    generative_parser.add_argument('x', type=str, help='生成に使用される入力データセット.'
                                                       'サポートしているファイルフォーマットは[.npy|.npz].')
    generative_parser.add_argument('save', type=str, help='結果を保存するディレクトリ.存在しない場合は終了する.'
                                                          'ファイル名は`generated.npz`となる.'
                                                          'すでに存在する場合はファイル名に索引を付ける.')
    generative_parser.add_argument('-b', '--batch', default=32, type=int, help='バッチサイズ.デフォルトは32.')
    generative_parser.set_defaults(func=command_generate)

    return parser.parse_args()

def main():
    from sys import exit, stderr
    args = parse_arg()
    try:
        args.func()
    except Exception as e:
        print(e, file=stderr)
        exit(1)

if __name__ == '__main__':
    main()

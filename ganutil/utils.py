import json
from os.path import isdir, isfile
from sys import stderr

import yaml
from keras.models import Sequential
from numpy import array_equal


def compile(discriminator, generator, dparameter, gparameter):
    """Ganの学習のために, Discriminatorモデルと[Generator + Discriminator]モデルをコンパイルする.

    :param keras.Model discriminator: Discriminatorモデル.
    :param keras.Model generator: Generatorモデル.
    :param dict dparameter: Discriminatorモデルのコンパイルに使用する引数.
    :param dict gparameter: [Generator + Discriminator]モデルのコンパイルに使用する引数.

    :return: コンパイル済みの[Generator + Discriminator]モデル,Discriminatorモデル,Generatorモデル.
    """
    discriminator.compile(**dparameter)
    set_trainability(discriminator, False)
    gan = Sequential((generator, discriminator))
    gan.compile(**gparameter)
    return gan, discriminator, generator


def save_architecture(dfilepath, gfilepath, discriminator, generator):
    for p, m in zip((dfilepath, gfilepath), (discriminator, generator)):
        with open(p, 'w') as f:
            extension = p.split('.')[-1]
            if extension == 'yml':
                f.write(m.to_yaml(indent=4))
            elif extension == 'json':
                f.write(m.to_json(indent=4))
            else:
                raise ValueError('Unknown file extension: ' + str(extension))


def save_dict(filepath, dictionary):
    extension = filepath.split('.')[-1]
    if extension == 'yml':
        yaml_mode = True
    elif extension == 'json':
        yaml_mode = False
    else:
        raise ValueError('Unknown file extension: ' + str(extension))
    with open(filepath, 'w') as f:
        if yaml_mode:
            f.write(yaml.dump(filepath, indent=4))
        else:
            f.write(json.dump(filepath, indent=4))


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def ensure_file(path, extensions=[]):
    """ファイルの存在を確認する.

    :param str paths: ファイルのパス.
    :param str extensions: ファイルが持つべき拡張子を指定する.
    :return bool: ファイルが確認できた場合のみTrue.それ以外はFalse.
    """
    if not isfile(path):
        print('%s is not valid.' % path, file=stderr)
        return False
    if len(extensions) > 0 and path.split('.')[-1] not in extensions:
        print('%s is not supported' % path)
        return False
    return True


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


def ensure_directory(path):
    """ディレクトリの存在を確認する.

    :param str path: ディレクトリのパス.
    :return bool: ディレクトリが確認できたらTrue.そうでなければFalse.
    """
    if not isdir(path):
        print('%s is not valid.' % path, file=stderr)
        return False
    return True


def ensure_input_shape(input_shape, data_shape, name):
    """モデルの入力層の形状とデータの形状が等しいことを確かめる.
    :param input_shape: モデルの入力層の形状.
    :param data_shape: データの形状.
    :param name: モデル名.
    :return: モデルの入力層の形状とデータの形状が等しいかどうか.
    """
    if not array_equal(input_shape, data_shape):
        print('input data shape %s does not equal to %s input layer shape %s' %
              (str(data_shape), name, str(input_shape)), file=stderr)
        return False
    return True

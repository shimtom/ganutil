from os.path import isdir, isfile
from sys import stderr

from numpy import array_equal


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

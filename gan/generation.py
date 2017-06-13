from numpy import array, savez
from os import join, isfile


def generate(generator, dataset, save, batch_size=32):
    """generatorを使用してデータセットから生成を行う.
    識別結果はsaveに指定されているディレクトリに`generated.npz`で保存される.
    ファイルがすでに存在する場合は索引を付与して保存する.

    :param keras.Model generator: generator モデル.
    :param numpy.ndarray dataset: データセット. 形状は[data size, generatorの入力形状].
    :param str save: 結果を保存するディレクトリ.
    :param int batch_size: バッチサイズ.
    """
    result = generator.predict(dataset, batch_size=batch_size)
    path = join(save, 'generated.npz')
    if isfile(path):
        index = 1
        while not isfile(join(save, 'generated(%d).npz' % index)):
            index += 1
        path = join(save, 'generated(%d).npz' % index)

    savez(path, array(result))

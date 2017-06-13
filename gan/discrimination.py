from numpy import array, savez
from os.path import join, isfile


def discriminate(discriminator, dataset, save, batch_size=32):
    """discriminatorを使用してデータセットに対して識別を行う.
    識別結果はsaverに指定されているディレクトリに`discriminated.npz`で保存される.
    ファイルがすでに存在する場合は索引を付与して保存する.

    :param keras.Model discriminator: discriminator モデル.
    :param numpy.ndarray dataset: データセット. 形状は[data size, discriminatorの入力形状].
    :param str save: 結果を保存するディレクトリ.
    :param int batch_size: バッチサイズ.
    """
    result = discriminator.predict(dataset, batch_size=batch_size)

    path = join(save, 'discriminated.npz')
    if isfile(path):
        index = 1
        while not isfile(join(save, 'discriminated(%d).npz' % index)):
            index += 1
        path = join(save, 'discriminated(%d).npz' % index)

    savez(path, array(result))

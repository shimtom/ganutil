from numpy import save, savez, savetxt


def discriminate(discriminator, dataset, save_path, batch_size=32):
    """discriminatorを使用してデータセットに対して識別を行い、結果を保存する.
    ファイルがすでに存在する場合は上書き保存を行う.

    :param keras.Model discriminator: discriminator モデル.
    :param numpy.ndarray dataset: データセット. 形状は[data size, discriminatorの入力形状].
    :param str save_path: 結果を保存するファイル名.
    :param int batch_size: バッチサイズ.
    """
    result = discriminator.predict(dataset, batch_size=batch_size)
    extension = save_path.split('.')[-1]
    if extension is 'npy' or save_path.find('.') == -1:
        save(save_path, result)
    elif extension is 'npz':
        savez(save_path, result)
    else:
        savetxt(save_path, result)

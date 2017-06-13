# Generative Adversarial Nets

GANモデルに対する操作をまとめたパッケージ.

## Install
```
$ python setup.py install
```

## Package Usage
* `gan.train(discriminator, generator, d_opt, g_opt, d_inputs, g_inputs, epoch_size, batch_size=32, preprocessor=default_preprocessor, saver=default_saver)`:

  GANを訓練する.

  * Arguments:
    - `discriminator`: keras.Model.discriminatorモデル.出力の形状は(data size, 1)で値は[0, 1]の範囲でなければならない.
    - `generator`: keras.Model. generatorモデル.出力の形状は(size, height, width, ch)で各値は[-1, 1]の範囲でなければならない.
    - `d_opt`: keras.Optimizer.discriminatorの学習に使用する最適化.
    - `g_opt`: keras.Optimizer.generatorの学習に使用する最適化.
    - `d_inputs`: numpy.ndarray.discriminatorの学習に使用する入力データセット.
    - `g_inputs`: numpy.ndarray.discriminatorの学習に使用する入力データセット.
    - `epoch_size`: int.最大のエポック数.
    - `batch_size`: int.バッチの大きさ.デフォルトは`32`.
    - `preprocessor`: keras.preprocessing.image.ImageDataGenerator.discriminatorの入力データに対して前処理を行ったデータのジェネレーター.デフォルトは何もしないジェネレーターを設定している.
    - `saver`: gan.Saver.各値を保存するセーバー.デフォルトは`save`ディレクトリに各値を保存する.

* `gan.discriminate(discriminator, dataset, save, batch_size=32)`

  discriminatorを使用してデータセットに対して識別を行う.
  識別結果はsaverに指定されているディレクトリに`discriminated.npz`で保存される.
  ファイルがすでに存在する場合は索引を付与して保存する.

  * Arguments
    - `discriminator`: keras.Model.discriminator モデル.
    - `dataset`: numpy.ndarray.データセット. 形状は[data size, discriminatorの入力形状].
    - `save`: str.結果を保存するディレクトリ.
    - `batch_size`: int.バッチサイズ.
* `gan.generate(generator, dataset, save, batch_size=32)`

  generatorを使用してデータセットから生成を行う.
  識別結果はsaveに指定されているディレクトリに`generated.npz`で保存される.
  ファイルがすでに存在する場合は索引を付与して保存する.

  * Arguments
    - `generator`: keras.Model.generator モデル.
    - `dataset`: numpy.ndarray.データセット. 形状は[data size, generatorの入力形状].
    - `save`: str.結果を保存するディレクトリ.
    - `batch_size`: int.バッチサイズ.


## Command Usage
```
usage: gan [-h] {train,discriminate,generate} ...

Generative Adversarial Nets.

positional arguments:
  {train,discriminate,generate}
    train               GANを訓練する.
    discriminate        学習済みのDiscriminatorモデルを用いて識別を行う.
    generate            学習済みのGeneratorモデルを用いて生成を行う.

optional arguments:
  -h, --help            show this help message and exit
```

* `train` command

  ```
  usage: gan train [-h] [--epoch EPOCH] [--batch BATCH] [--dweight DWEIGHT]
                      [--dlr DLR] [--dbeta1 DBETA1] [--gweight GWEIGHT]
                      [--glr GLR] [--gbeta1 GBETA1]
                      discriminator generator dinput ginput save

  positional arguments:
    discriminator      discriminatorモデル.サポートしているファイルフォーマットは[.json|.yml].kerasの仕様
                       に従ったものでなければならない.
    generator          generatorモデル.サポートしているファイルフォーマットは[.json|.yml].kerasの仕様に従った
                       ものでなければならない.
    dinput             discriminatorの訓練に使用する入力データセットのファイル名.サポートしているファイルフォーマットは[.
                       npy|.npz].
    ginput             generatorの訓練に使用する入力データセットのファイル名.サポートしているファイルフォーマットは[.npy|
                       .npz].
    save               学習結果を保存するディレクトリ.存在しない場合は終了する.

  optional arguments:
    -h, --help         show this help message and exit
    --epoch EPOCH      エポックサイズ.デフォルトは20回.
    --batch BATCH      バッチサイズ.デフォルトは32.
    --dweight DWEIGHT  discriminatorの学習済み重みパラメータ.指定したdiscriminatorのアーキテクチャでなければな
                       らない.サポートしているファイルフォーマットは[.h5].
    --dlr DLR          discriminatorの学習係数.
    --dbeta1 DBETA1    discriminatorのAdam Optimizerのbeta1の値.デフォルトは0.5.
    --gweight GWEIGHT  generatorの学習済み重みパラメータ.指定したgeneratorのアーキテクチャでなければならない.サポート
                       しているファイルフォーマットは[.h5].
    --glr GLR          generatorの学習係数.デフォルトは0.0002.
    --gbeta1 GBETA1    generatorのAdam Optimizerのbeta1の値.デフォルトは0.5.
  ```

* `generate` command

  ```
  usage: gan generate [-h] [-b BATCH] model x save

  positional arguments:
    model                 学習済みgeneratorモデル.サポートしているファイルフォーマットは[.h5].kerasの仕様に従った
                          ものでなければならない.
    x                     生成に使用される入力データセット.サポートしているファイルフォーマットは[.npy|.npz].
    save                  結果を保存するディレクトリ.存在しない場合は終了する.ファイル名は`generated.npz`となる.すで
                          に存在する場合はファイル名に索引を付ける.

  optional arguments:
    -h, --help            show this help message and exit
    -b BATCH, --batch BATCH
                          バッチサイズ.デフォルトは32.
  ```

* `discriminate` command

  ```
  usage: gan discriminate [-h] [-b BATCH] model x save

  positional arguments:
    model                 学習済みdiscriminatorモデル.サポートしているファイルフォーマットは[.h5].kerasの仕様
                          に従ったものでなければならない.
    x                     識別に使用されるデータセット.サポートしているファイルフォーマットは[.npy|.npz].
    save                  結果を保存するディレクトリ.存在しない場合は終了する.ファイル名は`discriminated.npz`とな
                          る.すでに存在する場合はファイル名に索引を付ける.

  optional arguments:
    -h, --help            show this help message and exit
    -b BATCH, --batch BATCH
                          バッチサイズ.デフォルトは32.
  ```

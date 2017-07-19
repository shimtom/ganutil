# Generative Adversarial Nets Utils

GANモデルに対する操作をまとめたパッケージ.

## Install
```
$ curl -O https://github.com/shimtom/ganutil/releases/download/ver0.3.1/ganutil-0.3.1-py3-none-any.whl
$ pip3 install ganutil-0.2.4-py3-none-any.whl
```

## Package Usage
* `ganutil.train(gan, discriminator, generator, d_inputs, g_inputs, epoch_size, batch_size=32, preprocessor=default_preprocessor, saver=default_saver)`:  
    GANを訓練する.  
    また,エポックごとに学習結果を保存する.それぞれの損失,精度のグラフ,モデル,パラメータ,生成画像が保存される.保存にはganutil.saverを使用する.  
    * Arguments:  
        - `gan`: keras.Model.compile済みgenerator + discriminatorモデル.generatorは訓練可能でなければならないがdiscriminatorは訓練可能であってはならない.
        - `discriminator`: keras.Model.compile済みdiscriminatorモデル.訓練可能でなければならない.出力の形状は(data size, 1)で値は[0, 1]の範囲でなければならない.
        - `generator`: keras.Model.ganに使用したgeneratorモデル.出力の形状は(size, height, width, ch)で各値は[-1, 1]の範囲でなければならない.
        - `d_inputs`: numpy.ndarray.discriminatorの学習に使用する入力データセット.
        - `g_inputs`: numpy.ndarray.discriminatorの学習に使用する入力データセット.
        - `epoch_size`: int.最大のエポック数.
        - `batch_size`: int.バッチの大きさ.デフォルトは`32`.
        - `preprocessor`: keras.preprocessing.image.ImageDataGenerator.discriminatorの入力データに対して前処理を行ったデータのジェネレーター.デフォルトは何もしないジェネレーターを設定している.
        - `saver`: ganutil.Saver.各値を保存するセーバー.デフォルトは`save`ディレクトリに各値を保存する.  

* `ganutil.discriminate(discriminator, dataset, save, batch_size=32)`  
    discriminatorを使用してデータセットに対して識別を行う.
    識別結果はsaverに指定されているディレクトリに`discriminated.npy`で保存される.
    ファイルがすでに存在する場合は索引を付与して保存する.

    * Arguments
        - `discriminator`: keras.Model.discriminator モデル.
        - `dataset`: numpy.ndarray.データセット. 形状は[data size, discriminatorの入力形状].
        - `save`: str.結果を保存するディレクトリ.
        - `batch_size`: int.バッチサイズ.


* `ganutil.generate(generator, dataset, save, batch_size=32)`
    generatorを使用してデータセットから生成を行う.
    識別結果はsaveに指定されているディレクトリに`generated.npy`で保存される.
    ファイルがすでに存在する場合は索引を付与して保存する.

    * Arguments
        - `generator`: keras.Model.generator モデル.
        - `dataset`: numpy.ndarray.データセット. 形状は[data size, generatorの入力形状].
        - `save`: str.結果を保存するディレクトリ.
        - `batch_size`: int.バッチサイズ.


## Command Usage
```
usage: ganutil [-h] {train,discriminate,generate} ...

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
    usage: ganutil train [-h] [--epoch EPOCH] [--batch BATCH] [--dweight DWEIGHT]
                      [--dlr DLR] [--dbeta1 DBETA1] [--gweight GWEIGHT]
                      [--glr GLR] [--gbeta1 GBETA1]
                      discriminator generator dinput ginput save

    positional arguments:
      discriminator      discriminatorモデル.サポートしているファイルフォーマットは[.json|.yml].kerasの仕様
                         に従ったものでなければならない.
      generator          generatorモデル.サポートしているファイルフォーマットは[.json|.yml].kerasの仕様に従った
                         ものでなければならない.
      dinput             discriminatorの訓練に使用する入力データセットのファイル名.サポートしているファイルフォーマットは[.
                         npy].
      ginput             generatorの訓練に使用する入力データセットのファイル名.サポートしているファイルフォーマットは[.npy].
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
    usage: ganutil generate [-h] [-b BATCH] model x save

    positional arguments:
      model                 学習済みgeneratorモデル.サポートしているファイルフォーマットは[.h5].kerasの仕様に従った
                            ものでなければならない.
      x                     生成に使用される入力データセット.サポートしているファイルフォーマットは[.npy].
      save                  結果を保存するファイルパス.拡張子がない場合は[.npy]で保存される.また,ディレクトリが存在しない場合は
                            終了し,ファイルがすでに存在する場合は上書きする.

    optional arguments:
      -h, --help            show this help message and exit
      -b BATCH, --batch BATCH
                            バッチサイズ.デフォルトは32.
    ```

* `discriminate` command

    ```
    usage: ganutil discriminate [-h] [-b BATCH] model x save

    positional arguments:
      model                 学習済みdiscriminatorモデル.サポートしているファイルフォーマットは[.h5].kerasの仕様
                            に従ったものでなければならない.
      x                     識別に使用されるデータセット.サポートしているファイルフォーマットは[.npy].
      save                  結果を保存するファイルパス.拡張子がない場合は[.npy]で保存される.また,ディレクトリが存在しない場合は
                            終了し,ファイルがすでに存在する場合は上書きする.

    optional arguments:
      -h, --help            show this help message and exit
      -b BATCH, --batch BATCH
                            バッチサイズ.デフォルトは32.
    ```

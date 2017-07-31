# Generative Adversarial Nets Utils

GANモデルに対する操作をまとめたパッケージ.

## Install
```
$ curl -O https://github.com/shimtom/ganutil/releases/download/ver0.4.1/ganutil-0.4.1-py3-none-any.whl
$ pip3 install ganutil-0.4.1-py3-none-any.whl
```

## Package Usage
### Gan class
`ganutil.Gan(generator, discriminator)`

### methods
* `compile(self, doptimizer, goptimizer, dloss, gloss, dmetrics=None, dloss_weights=None, dsample_weight_mode=None, gmetrics=None, gloss_weights=None, gsample_weight_mode=None)`
  Ganをcompileする.

  * Arguments
    - `doptimizer`: discriminatorのoptimizer
    - `goptimizer`: generatorのoptimizer
    - `dloss`: discriminatorのloss
    - `gloss`: generatorのloss
    - `dmetrics`: discriminatorのmetrics
    - `dloss_weights`: discriminatorのloss_weights
    - `dsample_weight_mode`: discriminatorのsample_weight_mode
    - `gmetrics`: generatorのmetrics
    - `gloss_weights`: generatorのloss_weights
    - `gsample_weight_mode`: generatorのsample_weight_mode


* `fit_generator(self, d_generator, g_generator, steps_per_epoch, d_iteration_per_step=1, g_iteration_per_step=1, epochs=1, d_callbacks=None, g_callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, initial_epoch=0)`
  Ganを訓練する.
  * Arguments
    - `d_generator`: discriminatorのデータジェネレーター.並列に処理される.
    - `g_generator`: generatorのデータジェネレーター.並列には処理されない.
    - `steps_per_epoch`: エポック毎のステップ数.
    - `d_iteration_per_step`: ステップ毎にdiscriminatorを学習する回数.
    - `g_iteration_per_step`: ステップ毎にgeneratorを学習する回数.
    - `epochs`: エポック数.
    - `d_callbacks`: discriminatorのコールバック.
    - `g_callbacks`: generatorのコールバック.
    - `max_queue_size`: キューの最大値.
    - `workers`: ワーカーの数.
    - `use_multiprocessing`: マルチプロセッシングを行うかどうか.
    - `initial_epoch`: 初期エポック.
  * Returns
    discriminatorのhistoryとgeneratorのhistory.

### functions
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

### callbacks
* `ganutil.callbacks.GeneratedImage(filepath, samples, normalize)`

* `ganutil.callbacks.ValueGraph(filepath, name, sample_mode='epoch')`

* `ganutil.callbacks.LossGraph(filepath, sample_mode='epoch')`

* `ganutil.callbacks.AccuracyGraph(filepath, sample_mode='epoch')`

* `ganutil.callbacks.ValueHistory(filepath, name, sample_mode='epoch')`

* `ganutil.callbacks.LossHistory(filepath, sample_mode='epoch')`

* `ganutil.callbacks.AccuracyHistory(filepath, sample_mode='epoch')`

* `ganutil.callbacks.GanProgbarLogger()`

* `ganutil.callbacks.GanModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)`

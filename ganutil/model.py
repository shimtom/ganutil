# -*- coding: utf-8 -*-
import warnings

import keras.callbacks as cbks
import tensorflow as tf
from keras.models import Sequential
from keras.utils import GeneratorEnqueuer, OrderedEnqueuer, Sequence

from .callbacks import GanProgbarLogger


class Gan(object):
    """Ganモデルクラス."""
    def __init__(self, generator, discriminator):
        """
        :param generator: Generatorモデル.
        :param discriminator: Discriminatorモデル.
        """
        self.discriminator = discriminator
        # only generator model
        self.generator_model = generator
        # generator + freeze discriminator model
        self.generator = None

    def compile(self, doptimizer, goptimizer, dloss, gloss,
                dmetrics=None, dloss_weights=None, dsample_weight_mode=None,
                gmetrics=None, gloss_weights=None, gsample_weight_mode=None):
        """
        Ganをcompileする.

        :param doptimizer: discriminatorのoptimizer
        :param goptimizer: generatorのoptimizer
        :param dloss: discriminatorのloss
        :param gloss: generatorのloss
        :param dmetrics: discriminatorのmetrics
        :param dloss_weights: discriminatorのloss_weights
        :param dsample_weight_mode: discriminatorのsample_weight_mode
        :param gmetrics: generatorのmetrics
        :param gloss_weights: generatorのloss_weights
        :param gsample_weight_mode: generatorのsample_weight_mode
        """

        self.generator_model._make_predict_function()
        self.generator_graph = tf.get_default_graph()
        self.discriminator.compile(doptimizer, dloss, metrics=dmetrics,
                                   loss_weights=dloss_weights,
                                   sample_weight_mode=dsample_weight_mode)
        self.discriminator.trainable = False

        self.generator = Sequential((self.generator_model, self.discriminator))
        self.generator.compile(goptimizer, gloss, metrics=gmetrics,
                               loss_weights=gloss_weights,
                               sample_weight_mode=gsample_weight_mode)

    def fit_generator(self, d_generator, g_generator, steps_per_epoch,
                      d_iteration_per_step=1, g_iteration_per_step=1, epochs=1,
                      d_callbacks=None, g_callbacks=None, max_queue_size=10,
                      workers=1, use_multiprocessing=False,
                      initial_epoch=0):
        """
        Ganを訓練する.

        :param d_generator: discriminatorのデータジェネレーター.並列に処理される.
        :param g_generator: generatorのデータジェネレーター.並列には処理されない.
        :param steps_per_epoch: エポック毎のステップ数.
        :param d_iteration_per_step: ステップ毎にdiscriminatorを学習する回数.
        :param g_iteration_per_step: ステップ毎にgeneratorを学習する回数.
        :param epochs: エポック数.
        :param d_callbacks: discriminatorのコールバック.
        :param g_callbacks: generatorのコールバック.
        :param max_queue_size: キューの最大値.
        :param workers: ワーカーの数.
        :param use_multiprocessing: マルチプロセッシングを行うかどうか.
        :param initial_epoch: 初期エポック.
        :return: discriminatorのhistoryとgeneratorのhistory.
        """
        # FIXME: fix problem that processes stop in `use_multiprocessing=True`
        if use_multiprocessing:
            warnings.warn('Multi-process will not be working.')

        d_is_sequence = isinstance(d_generator, Sequence)
        g_is_sequence = isinstance(g_generator, Sequence)

        if not (d_is_sequence and g_is_sequence) and use_multiprocessing and workers > 1:
            warnings.warn('Using a generator with `use_multiprocessing=True`'
                          ' and multiple workers may duplicate your data.'
                          ' Please consider using the`keras.utils.Sequence'
                          ' class.')
        d_enqueuer = None
        g_enqueuer = None

        wait_time = 0.01  # in seconds

        try:
            # TODO: データジェネレーター内部でpredict()を使用した時に
            # 上手くいかない問題を修正する
            # if d_is_sequence:
            #     d_enqueuer = OrderedEnqueuer(
            #         d_generator, use_multiprocessing=use_multiprocessing)
            # else:
            #     d_enqueuer = GeneratorEnqueuer(
            #         d_generator, use_multiprocessing=use_multiprocessing, wait_time=wait_time)
            # d_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            # d_sample_generator = d_enqueuer.get()
            d_sample_generator = d_generator

            if g_is_sequence:
                g_enqueuer = OrderedEnqueuer(
                    g_generator, use_multiprocessing=use_multiprocessing)
            else:
                g_enqueuer = GeneratorEnqueuer(
                    g_generator, use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            g_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            g_sample_generator = g_enqueuer.get()

            self.discriminator.history = cbks.History()
            # BaseLoggerは1番目でなければならない
            d_callbacks = [cbks.BaseLogger()] + (d_callbacks or [])
            d_callbacks += [self.discriminator.history]
            for c in d_callbacks:
                if isinstance(c, cbks.ProgbarLogger):
                    warnings.warn('Using a `keras.callbacks.ProgbarLogger`, '
                                  ' it can\'t distinguishe whether output is '
                                  'generator\'s or discriminator\'s. Please '
                                  'consider using the `ganutil.callbacks.'
                                  'ProgbarLogger` class.')

            d_callbacks = cbks.CallbackList(d_callbacks)
            d_callbacks.set_model(self.discriminator)
            d_callbacks.set_params({
                'epochs': epochs,
                'steps': steps_per_epoch,
                'verbose': 1,
                'do_validation': False,
                'metrics': self.discriminator.metrics_names,
            })

            self.generator.history = cbks.History()
            # BaseLoggerは1番目でなければならない
            g_callbacks = [cbks.BaseLogger()] + (g_callbacks or [])
            g_callbacks += [self.generator.history]
            for c in g_callbacks:
                if isinstance(c, cbks.ProgbarLogger):
                    warnings.warn('Using a `keras.callbacks.ProgbarLogger, `'
                                  ' it can\'t distinguishe whether output is '
                                  'generator\'s or discriminator\'s. Please'
                                  ' consider using the`ganutil.callbacks.'
                                  'ProgbarLogger` class')
                if isinstance(c, cbks.ModelCheckpoint):
                    warnings.warn('Using a `keras.callbacks.ModelCheckpoint, `'
                                  ' it can\'t save only generator model.'
                                  ' Please consider using the'
                                  ' `ganutil.callbacks.GanModelCheckpoint'
                                  ' class.')

            g_callbacks = cbks.CallbackList(g_callbacks)
            g_callbacks.set_model(self.generator)
            g_callbacks.set_params({
                'epochs': epochs,
                'steps': steps_per_epoch,
                'verbose': 1,
                'do_validation': False,
                'metrics': self.generator.metrics_names,
            })

            # BaseLoggerは1番目でなければならない
            common_callbacks = [GanProgbarLogger()]
            common_callbacks = cbks.CallbackList(common_callbacks)
            common_callbacks.set_model(self.generator)
            common_callbacks.set_params({
                'epochs': epochs,
                'steps': steps_per_epoch,
                'verbose': 1,
                'do_validation': False,
                'metrics': {
                    'discriminator': self.discriminator.metrics,
                    'generator': self.generator.metrics_names,
                }
            })

            d_callbacks.on_train_begin()
            g_callbacks.on_train_begin()
            common_callbacks.on_train_begin()

            d_epoch_logs = {}
            g_epoch_logs = {}
            common_epoch_logs = {
                'discriminator': d_epoch_logs,
                'generator': g_epoch_logs,
            }
            for epoch in range(initial_epoch, epochs):
                d_callbacks.on_epoch_begin(epoch, d_epoch_logs)
                g_callbacks.on_epoch_begin(epoch, g_epoch_logs)
                common_callbacks.on_epoch_begin(epoch, common_epoch_logs)
                for step in range(steps_per_epoch):
                    d_batch_logs = {}
                    g_batch_logs = {}
                    common_batch_logs = {
                        'discriminator': d_batch_logs,
                        'generator': g_batch_logs,
                    }

                    common_callbacks.on_batch_begin(step, common_batch_logs)

                    for index in range(d_iteration_per_step):
                        samples = next(d_sample_generator)
                        d_batch_logs['batch'] = step
                        d_batch_logs['iteration'] = index
                        d_batch_logs['size'] = samples[0].shape[0]
                        d_callbacks.on_batch_begin(step, d_batch_logs)
                        d_outs = self.discriminator.train_on_batch(*samples)
                        if not isinstance(d_outs, list):
                            d_outs = [d_outs]
                        for n, o in zip(self.discriminator.metrics_names, d_outs):
                            d_batch_logs[n] = o
                        d_callbacks.on_batch_end(step, d_batch_logs)

                    for index in range(g_iteration_per_step):
                        samples = next(g_sample_generator)
                        g_batch_logs['batch'] = step
                        g_batch_logs['iteration'] = index
                        g_batch_logs['size'] = samples[0].shape[0]
                        g_callbacks.on_batch_begin(step, g_batch_logs)
                        g_outs = self.generator.train_on_batch(*samples)
                        if not isinstance(g_outs, list):
                            g_outs = [g_outs]
                        for n, o in zip(self.generator.metrics_names, g_outs):
                            g_batch_logs[n] = o
                        g_callbacks.on_batch_end(step, g_batch_logs)

                    common_callbacks.on_batch_end(step, common_batch_logs)

                d_callbacks.on_epoch_end(epoch, d_epoch_logs)
                g_callbacks.on_epoch_end(epoch, g_epoch_logs)
                common_callbacks.on_epoch_end(epoch, common_epoch_logs)

            d_callbacks.on_train_end()
            g_callbacks.on_train_end()
            common_callbacks.on_train_end()

        finally:
            if d_enqueuer is not None:
                d_enqueuer.stop()
            if g_enqueuer is not None:
                g_enqueuer.stop()

        return self.discriminator.history, self.generator.history

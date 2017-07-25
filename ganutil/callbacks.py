# -*- coding: utf-8 -*-
import math
import os

import keras.callbacks as cbks
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.utils import Progbar


class GeneratedImage(cbks.Callback):
    def __init__(self, filepath, samples, normalize):
        super(GeneratedImage, self).__init__()

        self.filepath = filepath
        self.samples = samples
        self.normalize = normalize

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if not os.path.isdir(os.path.dirname(filepath)):
            os.mkdir(os.path.dirname(filepath))

        generator = self.model.layers[0]
        images = self.normalize(generator.predict_on_batch(self.samples))
        if len(images.shape) == 4 and images.shape[-1] == 1:
            images = images.reshape(images.shape[:-1])

        columns = int(math.sqrt(len(images)))
        rows = int(len(images) // columns)
        plt.figure()
        for i, image in enumerate(images):
            if i < columns * rows:
                plt.subplot(columns, rows, i + 1)
                plt.imshow(image)
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()


class ValueGraph(cbks.Callback):
    def __init__(self, filepath, name, sample_mode='epoch'):
        super(ValueGraph, self).__init__()
        self.filepath = filepath
        self.name = name

        if sample_mode == 'epoch':
            self.epoch_mode = True
        elif sample_mode == 'batch':
            self.epoch_mode = False
        else:
            raise ValueError('Unknown `sample_mode`: ' + str(sample_mode))

    def on_train_begin(self, logs={}):
        self.values = []
        self.epoch_values = []
        self.total_value = 0.
        self.total_size = 0

    def on_batch_end(self, batch, logs={}):
        value = logs.get(self.name, 0.)
        batch_size = logs.get('size', 0)

        self.values.append(value)
        self.total_value += value * batch_size
        self.total_size += batch_size

        if not self.epoch_mode:
            filepath = self.filepath.format(
                batch=batch, name=self.name, **logs)
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
            self._plot(filepath, self.values)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_values.append(self.total_value / self.total_size)
        self.total_value = 0.
        self.total_size = 0.

        if self.epoch_mode:
            filepath = self.filepath.format(
                epoch=epoch, name=self.name, **logs)
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
            self._plot(filepath, self.epoch_values)

    def _plot(filepath, values):
        sns.set(style='darkgrid', palette='deep', color_codes=True)
        plt.figure()
        plt.plot(values)
        plt.xlim([0, len(values)])
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()


class LossGraph(ValueGraph):
    def __init__(self, filepath, sample_mode='epoch'):
        super(LossGraph, self).__init__(filepath, 'loss', sample_mode)


class AccuracyGraph(ValueGraph):
    def __init__(self, filepath, sample_mode='epoch'):
        super(AccuracyGraph, self).__init__(filepath, 'acc', sample_mode)


class ValueHistory(cbks.Callback):
    def __init__(self, filepath, name, sample_mode='epoch'):
        super(ValueHistory, self).__init__()
        self.filepath = filepath
        self.name = name

        if sample_mode == 'epoch':
            self.epoch_mode = True
        elif sample_mode == 'batch':
            self.epoch_mode = False
        else:
            raise ValueError('Unknown `sample_mode`: ' + str(sample_mode))

    def on_train_begin(self, logs={}):
        self.values = []
        self.epoch_values = []
        self.total_value = 0.
        self.total_size = 0

    def on_batch_end(self, batch, logs={}):
        value = logs.get(self.name, 0.)
        batch_size = logs.get('size', 0)

        self.values.append(value)
        self.total_value += value * batch_size
        self.total_size += batch_size

        if not self.epoch_mode:
            filepath = self.filepath.format(
                batch=batch, name=self.name, **logs)
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
            np.save(filepath, np.array(self.values))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_values.append(self.total_value / self.total_size)
        self.total_value = 0.
        self.total_size = 0.

        if self.epoch_mode:
            filepath = self.filepath.format(
                epoch=epoch, name=self.name, **logs)
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
            np.save(filepath, np.array(self.epoch_values))


class LossHistory(ValueHistory):
    def __init__(self, filepath, sample_mode='epoch'):
        super(LossHistory, self).__init__(filepath, 'loss', sample_mode)


class AccuracyHistory(ValueHistory):
    def __init__(self, filepath, sample_mode='epoch'):
        super(AccuracyHistory, self).__init__(filepath, 'acc', sample_mode)


class ProgbarLogger(cbks.ProgbarLogger):
    def __init__(self, name, count_mode='samples'):
        super(ProgbarLogger, self).__init__(count_mode=count_mode)
        self.name = name

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            print('%s Epoch %d/%d' % (self.name, epoch + 1, self.epochs))
            if self.use_steps:
                target = self.params['steps']
            else:
                target = self.params['samples']
            self.target = target
            self.progbar = Progbar(target=self.target, verbose=self.verbose)
        self.seen = 0


class GanModelCheckpoint(cbks.ModelCheckpoint):
    """only use as generator Callback"""

    def on_train_begin(self):
        self.model = self.model.layers[0]

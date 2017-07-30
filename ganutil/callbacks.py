# -*- coding: utf-8 -*-
import math
import os
import sys
import time

import keras.callbacks as cbks
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




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
        images = np.array(images).astype(np.uint8)
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

    def on_batch_end(self, batch, logs={}):
        self.values.append(logs.get(self.name, 0.))

        if not self.epoch_mode:
            filepath = self.filepath.format(
                batch=batch, name=self.name, **logs)
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
            self._plot(filepath, self.values)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_values.append(logs.get(self.name, 0.))

        if self.epoch_mode:
            filepath = self.filepath.format(
                epoch=epoch, name=self.name, **logs)
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
            self._plot(filepath, self.epoch_values)

    def _plot(self, filepath, values):
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

    def on_batch_end(self, batch, logs={}):
        self.values.append(logs.get(self.name, 0.))

        if not self.epoch_mode:
            filepath = self.filepath.format(
                batch=batch, name=self.name, **logs)
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
            np.save(filepath, np.array(self.values))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_values.append(logs.get(self.name, 0.))

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


class GanProgbar:
    def __init__(self, total,  width=30, interval=0.05):
        self.start = time.time()
        self.width = width
        self.interval = interval
        self.total = total
        self.numdigits = int(math.floor(math.log10(self.total))) + 1
        self.counter = '%%%dd/%%%dd ' % (self.numdigits, self.numdigits)
        self.total_width = 0
        pass

    def update(self,  count, dlogs, glogs):
        prev_total_width = self.total_width
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')

        bar = self.counter % (count, self.total)

        if self.total is not -1:
            prog = float(count) / self.total
            prog_width = int(self.width * prog)
            bar += '['
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                bar += '>' if count < self.total else '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)
        info = ' %ds' % (time.time() - self.start)
        info += ' D::'
        for k, v in dlogs.items():
            if abs(v) > 1e-3:
                info += ' - %s: %.4f' % (k, v)
            else:
                info += ' - %s: %.4e' % (k, v)

        info += ' G::'
        for k, v in glogs.items():
            if abs(v) > 1e-3:
                info += ' - %s: %.4f' % (k, v)
            else:
                info += ' - %s: %.4e' % (k, v)
        sys.stdout.write(info)
        self.total_width += len(info)

        if count >= self.total:
            sys.stdout.write('\n')


class GanProgbarLogger(cbks.Callback):
    def on_train_begin(self, logs={}):
        self.progbar = GanProgbar(self.params.get('epochs', -1))

    def on_epoch_end(self, epoch, logs={}):
        dlogs = {}
        for k in self.params['metrics']['discriminator']:
            if k in logs['discriminator']:
                dlogs[k] = logs['discriminator'][k]
        glogs = {}
        for k in self.params['metrics']['generator']:
            if k in logs['generator']:
                glogs[k] = logs['generator'][k]

        self.progbar.update(epoch, dlogs, glogs)


class ProgLogger(cbks.Callback):
    def __init__(self, name):
        super(ProgLogger, self).__init__()
        self.name = name

    def on_train_begin(self, logs={}):
        self.epochs = self.params.get('epochs', -1)

    def on_epoch_begin(self, epoch, logs={}):
        self.steps = self.params.get('steps', -1)
        self.epoch = epoch
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs={}):
        self.batch_start = time.time()

    def on_batch_end(self, batch, logs={}):
        elapsed_sec = math.ceil(time.time() - self.batch_start)
        info = 'Epoch %d/%d - %s' % (self.epoch, self.epochs, self.name)
        info += ' [%d/%d] - %ds' % (batch, self.steps, elapsed_sec)

        for k in self.params['metrics']:
            if k in logs:
                info += ' - %s: %.4f' % (k, logs[k])

        print(info)

    def on_epoch_end(self, epoch, logs={}):
        elapsed_sec = math.ceil(time.time() - self.epoch_start)
        info = 'Epoch %d/%d - %s' % (epoch, self.epochs, self.name)
        info += ' - %ds' % (elapsed_sec)

        for k in self.params['metrics']:
            if k in logs:
                info += ' - %s: %.4f' % (k, logs[k])

        print(info)


class GanModelCheckpoint(cbks.ModelCheckpoint):
    """only use as generator Callback"""

    def on_train_begin(self, logs={}):
        self.model = self.model.layers[0]

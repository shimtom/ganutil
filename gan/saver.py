from os.path import join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class Saver:
    """学習の各値を保存する機能を持ったクラス.
    train()でのみ使用される.
    """
    def __init__(self, root, name='GAN'):
        """
        :param str root: 保存ディレクトリ.
        :param str name: Saverの名前.デフォルトは`GAN`.
        """
        self._root = root
        self._name = name
        self._generator = _Saver('generator', root)
        self._discriminator = _Saver('discriminator', root)
        self._loss_dir = join(root, 'losses')
        self._accuracy_dir = join(root, 'accuracy')
        self._image_dir = join(root, 'image')

        ensure_directory(self._root)
        ensure_directory(self._loss_dir)
        ensure_directory(self._accuracy_dir)
        ensure_directory(self._image_dir)

    @property
    def root(self):
        return self._root

    def config(self, config):
        import yaml
        with open(join(self._root, 'config.yml'), 'w') as f:
            f.write(yaml.dump(config, indent=4))

    def architecture(self, discriminator, generator):
        with open(join(self._root, 'discriminator.yml'), 'w') as f:
            f.write(discriminator.to_yaml(indent=4))
        with open(join(self._root, 'generator.yml'), 'w') as f:
            f.write(generator.to_yaml(indent=4))

    def model(self, discriminator, generator):
        discriminator.save(join(self._root, 'discriminator.h5'))
        generator.save(join(self._root, 'generator.h5'))

    def parameter(self, discriminator, generator):
        discriminator.save_weights(join(self._root, 'discriminator_weight.h5'))
        generator.save_weights(join(self._root, 'generator_weight.h5'))

    def result(self, arr, compression=False):
        path = join(self._root, '%s-result' % self._name)
        if compression:
            np.savez(path, arr)
            return
        np.save(path, arr)

    def loss(self, discriminator_loss, generator_loss):
        self._generator.loss(generator_loss)
        self._discriminator.loss(discriminator_loss)

        sns.set(style='darkgrid', palette='deep', color_codes=True)
        plt.figure()
        plt.plot(generator_loss, label='generator loss')
        plt.plot(discriminator_loss, label='discriminator loss')
        plt.xlim([0, max(len(generator_loss), len(discriminator_loss))])
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(self._loss_dir, 'loss.png'))
        plt.close()

    def accuracy(self, discriminator_accuracy, generator_accuracy):
        self._generator.accuracy(generator_accuracy)
        self._discriminator.accuracy(discriminator_accuracy)

        sns.set(style='darkgrid', palette='deep', color_codes=True)
        plt.figure()
        plt.plot(generator_accuracy, label='generator loss')
        plt.plot(discriminator_accuracy, label='discriminator loss')
        plt.xlim([0, max(len(generator_accuracy), len(discriminator_accuracy))])
        plt.ylim([0, 1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(self._accuracy_dir, 'accuracy.png'))
        plt.close()

    def image(self, images, id=None):
        from math import sqrt

        path = self._image_dir
        if id is not None:
            path = join(self._image_dir, '%03d' % id)
        ensure_directory(path)

        columns = int(sqrt(len(images)))
        rows = int(len(images) // columns)

        if images.shape[-1] == 1:
            images = images.reshape(images.shape[:-1])

        plt.figure()
        for i, img in enumerate(images):
            if i < columns * rows:
                plt.subplot(columns, rows, i + 1)
                plt.imshow(img)
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(join(path, 'image.png'))
        plt.close()

class _Saver:
    def __init__(self, name, root):
        self._name = name
        self._root = root
        self._loss_dir = join(root, 'losses')
        self._accuracy_dir = join(root, 'accuracy')

        ensure_directory(self._root)
        ensure_directory(self._loss_dir)
        ensure_directory(self._accuracy_dir)

    def summary(self, summary):
        with open(join(self._root, '%s.txt' % self._name), 'w') as f:
            f.write(summary)

    def loss(self, losses):
        np.save(join(self._loss_dir, self._name), losses)

        sns.set(style='darkgrid', palette='deep', color_codes=True)
        plt.figure()
        plt.plot(losses)
        plt.xlim([0, len(losses)])
        plt.title(self._name)
        plt.savefig(join(self._loss_dir, '%s.png' % self._name))
        plt.close()

    def accuracy(self, accuracies):
        np.save(join(self._accuracy_dir, self._name), accuracies)

        sns.set(style='darkgrid', palette='deep', color_codes=True)
        plt.figure()
        plt.plot(accuracies)
        plt.xlim([0, len(accuracies)])
        plt.ylim([0, 1])
        plt.title(self._name)
        plt.savefig(join(self._accuracy_dir, '%s.png' % self._name))
        plt.close()


def ensure_directory(path):
    import os
    if not os.path.isdir(path):
        os.makedirs(path)

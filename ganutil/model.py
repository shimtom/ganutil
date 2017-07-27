import tensorflow as tf
from keras.models import Sequential


class Gan(object):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, doptimizer, goptimizer, dloss, gloss,
                dmetrics=None, dloss_weights=None, dsample_weight_mode=None,
                gmetrics=None, gloss_weights=None, gsample_weight_mode=None):
        self.generator._make_predict_function()
        self.generator_graph = tf.get_default_graph()
        self.discriminator.compile(doptimizer, dloss, metrics=dloss_weights,
                                   loss_weights=dloss_weights,
                                   sample_weight_mode=dsample_weight_mode)
        self.discriminator.trainable = False

        self.gan = Sequential((self.discriminator, self.generator))
        self.gan.compile(goptimizer, gloss, metrics=gmetrics,
                         loss_weights=gloss_weights,
                         gsample_weight_mode=gsample_weight_mode)

import json

import yaml
from keras.models import Sequential


def compile(discriminator, generator, dparameter, gparameter):
    """Ganの学習のために, Discriminatorモデルと[Generator + Discriminator]モデルをコンパイルする.

    :param keras.Model discriminator: Discriminatorモデル.
    :param keras.Model generator: Generatorモデル.
    :param dict dparameter: Discriminatorモデルのコンパイルに使用する引数.
    :param dict gparameter: [Generator + Discriminator]モデルのコンパイルに使用する引数.

    :return: コンパイル済みの[Generator + Discriminator]モデル,Discriminatorモデル,Generatorモデル.
    """
    discriminator.compile(**dparameter)
    set_trainability(discriminator, False)
    gan = Sequential((generator, discriminator))
    gan.compile(**gparameter)
    return gan, discriminator, generator


def save_architecture(dfilepath, gfilepath, discriminator, generator):
    for p, m in zip((dfilepath, gfilepath), (discriminator, generator)):
        with open(p, 'w') as f:
            extension = p.split('.')[-1]
            if extension == 'yml':
                f.write(m.to_yaml(indent=4))
            elif extension == 'json':
                f.write(m.to_json(indent=4))
            else:
                raise ValueError('Unknown file extension: ' + str(extension))


def save_dict(filepath, dictionary):
    extension = filepath.split('.')[-1]
    if extension == 'yml':
        yaml_mode = True
    elif extension == 'json':
        yaml_mode = False
    else:
        raise ValueError('Unknown file extension: ' + str(extension))
    with open(filepath, 'w') as f:
        if yaml_mode:
            f.write(yaml.dump(filepath, indent=4))
        else:
            f.write(json.dump(filepath, indent=4))


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

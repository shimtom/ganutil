import json
import os
import tempfile

import ganutil.utils as utils
import pytest
import yaml
from keras.layers import Dense
from keras.models import Sequential


@pytest.fixture()
def dirpath():
    tempdir = tempfile.TemporaryDirectory()
    print('Directory: %s' % tempdir.name)
    yield tempdir.name
    print('Directory clean upped.')
    tempdir.cleanup()


@pytest.mark.util
@pytest.mark.parametrize('de, ge, dload, gload', [
    ('.yml', '.yml', yaml.load, yaml.load),
    ('.json', '.yml', json.load, yaml.load),
    ('.yml', '.json', yaml.load, json.load),
    ('.json', '.json', json.load, json.load),
])
def test_save_architecture(dirpath, de, ge, dload, gload):
    discriminator = Sequential()
    discriminator.add(Dense(1, input_shape=(16,)))
    generator = Sequential()
    generator.add(Dense(16, input_shape=(32,)))

    dpath = os.path.join(dirpath, 'discriminator')
    gpath = os.path.join(dirpath, 'generator')

    utils.save_architecture(dpath + de, gpath + ge, discriminator, generator)
    with open(dpath + de) as d:
        dload(d)

    with open(gpath + ge) as g:
        gload(g)

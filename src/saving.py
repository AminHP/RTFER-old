# -*- coding: utf-8 -*-

# python imports
import h5py
import numpy as np

# keras imports
import keras


def load_model(filepath):
    model = keras.models.load_model(filepath)
    f = h5py.File(filepath, mode='r')
    model.labels = [s.decode('utf-8') for s in f.attrs['_labels']]
    f.close()
    return model


def save_model(model, filepath, overwrite=True, save_weights_only=False):
    if save_weights_only:
        model.save_weights(filepath, overwrite=overwrite)
    else:
        model.save(filepath, overwrite=overwrite)

    f = h5py.File(filepath, mode='a')
    f.attrs['_labels'] = np.array(model.labels, dtype='S')
    f.close()

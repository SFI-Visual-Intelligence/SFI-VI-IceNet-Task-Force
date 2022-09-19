"""
Description: Not important for the project. Only personal file to play around with the code.
"""
import tensorflow as tf
from models import unet_batchnorm
import config
import os
import numpy as np


def get_dataloader():
    dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
    dataloader_config_fpath = os.path.join(
        config.dataloader_config_folder, dataloader_ID + ".json"
    )


def dropout_monte_carlo(model, x, n_iter: int=10):
    shape = (n_iter,) + x.shape
    dropout_ensemble = np.empty(shape)
    for i in range(n_iter):
        tf.random.set_seed(i)
        dropout_ensemble[i, ...] = model(x)
        
    ## get statistics
    print(dropout_ensemble.mean(0))
    print(dropout_ensemble.std(0))
    return dropout_ensemble.mean(0), dropout_ensemble.std(0)


if __name__ == "__main__":
    # model = unet_batchnorm()
    model = tf.keras.layers.Dropout(0.9, input_shape=(2,))
    x = np.arange(10).reshape(5,2).astype(np.float32)
    dropout_monte_carlo(model, x)

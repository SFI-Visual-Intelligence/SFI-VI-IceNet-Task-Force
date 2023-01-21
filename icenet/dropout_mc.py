"""
Description: Not important for the project. Only personal file to play around with the code.
"""
from typing import Iterable
import tensorflow as tf
from models import unet_batchnorm
import config
import os
import numpy as np

# from permute_and_predict import init_dates, network
# from models import DropoutWDefaultTraining


def get_dataloader():
    dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
    dataloader_config_fpath = os.path.join(
        config.dataloader_config_folder, dataloader_ID + ".json"
    )


def dropout_monte_carlo(network, x, seeds: Iterable[int] = None) -> np.ndarray:
    """
    Run dropout monte carlo on a network.

    Parameters
    ----------
    network : tf.keras.Model
        Network to run dropout monte carlo on.
    x : np.ndarray
        Input data.
    seeds : Iterable[int], optional
        Seeds to use for dropout monte carlo, by default: 0 trough 24 (range(25)).
    """
    if seeds is None:
        seeds = range(25)

    dropout_ensemble = []
    for seed in seeds:
        tf.random.set_seed(seed)
        dropout_ensemble.append(network.predict(x)[0])
    dropout_ensemble = np.array(dropout_ensemble)
    # ## get statistics
    # print(dropout_ensemble.mean(0))
    # print(dropout_ensemble.std(0))
    # return dropout_ensemble.mean(0), dropout_ensemble.std(0)
    return dropout_ensemble


def run_icenet_dmc_on_numpy(
    input_arr, ensemble, batch_size=8, seeds: Iterable[int] = tuple(range(10))
):
    """
    Run icenet drop out monte carlo on numpy data. Fuction adapted from original run_icenet_on_numpy function.

    Computes a numpy array of IceNet drop-out monte carlo ensemble-mean ice class forecasts using an
    numpy array of input tensors.

    Inputs:
    input_arr (np.ndarray): Shape (n_forecast_start_dates, n_x, n_y, n_input_vars).
    model: TensorFlow network (with dropout implemented) to use for computing ensamble mean over for several seeds.
    batch_size (int): Number of samples to predict in parallel on the GPU.
    seeds: Seeds corresponding to the model dropout variations that will make up the ensemble.

    Returns:
    all_icenet_preds (np.ndarray): Shape (n_forecast_start_dates, n_x, n_y,
    n_classes, n_leadtimes) of ice class index prediciont (0, 1, or 2).
    """

    num_batches = int(np.ceil(input_arr.shape[0] / batch_size))

    all_icenet_preds = []
    for batch_idx in range(num_batches):

        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, len(init_dates)])
        inputs = input_arr[batch_start:batch_end]

        network_preds = []
        network_preds = dropout_monte_carlo(network, inputs, seeds=seeds)
        # for network in ensemble:
        #     network_preds.append(network.predict(inputs))

        all_icenet_preds.append(np.array(network_preds))

    all_icenet_preds = np.mean(np.concatenate(all_icenet_preds, axis=1), axis=0)
    all_icenet_preds = all_icenet_preds.argmax(axis=-2).astype(int)

    return all_icenet_preds


if __name__ == "__main__":
    # model = unet_batchnorm()
    model = tf.keras.layers.Dropout(0.9, input_shape=(2,))
    x = np.arange(10).reshape(5, 2).astype(np.float32)
    dropout_monte_carlo(model, x)

"""
Tensorflow implementation of Guided Backpropagation. 
source: https://colab.research.google.com/drive/17tAC7xx2IJxjK700bdaLatTVeDA02GJn?usp=sharing#scrollTo=4-OduFD-wH14
"""

import tensorflow as tf

# from models import unet_batchnorm_w_dropout
from tensorflow.keras.applications.resnet import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from warnings import warn
from scipy import ndimage

## For managing configurations
import hydra
from omegaconf import DictConfig, OmegaConf

# from einops import repeat

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

## To load the model (with its custom parts)
from utils.models import (
    unet_batchnorm_w_dropout_mod,
    DropoutWDefaultTraining,
    construct_sum_last_two_channels_loss,
    ConstructLeadtimeAccuracy,
    TemperatureScale,
)
import os

# import cv2
import re


MODEL_PATH = "/storage/experiments/icenet/icenet/analyze_input_influences/test_data/network_dropout_mc_50.h5"
LAND_MASK_PATH = (
    "/storage/experiments/icenet/icenet/analyze_input_influences/land_mask.npy"
)
REGION_MASK_PATH = (
    "/storage/experiments/icenet/icenet/analyze_input_influences/region_mask.npy"
)
ORDERED_OBS_NPZS_DIR = (
    "/storage/experiments/icenet/data/ordered_obs_npz/yearly_samples/"
)

# MODEL_PATH = "/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper/icenet/analyze_input_influences/test_data/network_dropout_mc_50.h5"
# LAND_MASK_PATH = "/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper/data/masks/land_mask.npy"
# REGION_MASK_PATH = "/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper/data/masks/region_mask.npy"
# ORDERED_OBS_NPZS_DIR = "/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper/data/ordered_obs_npz/yearly_samples/"

DATA_PATHS = [
    os.path.join(ORDERED_OBS_NPZS_DIR, x) for x in os.listdir(ORDERED_OBS_NPZS_DIR)
]


def get_datafilename_by_year(year, data_paths: list) -> str:
    """
    Matches the year to the correct data file name and returns that file.
    Note that the function assumes there only is one file per year.

        Args:
            year (int): The year to match to a data file.
            data_paths (list): List of paths to the data files.

        Returns:
            str: The path to the data file.
    """
    pattern = re.compile(f".*{year}.*")
    for p in data_paths:
        if pattern.match(p):
            return p


@hydra.main(config_path="config", config_name="config.yaml")
def get_pred_and_explanations(cfg, leadtime) -> tuple:
    land_mask = np.load(LAND_MASK_PATH)
    hudson_bay_mask = np.load(REGION_MASK_PATH) == 5

    data_path = get_datafilename_by_year(cfg.predict_year, DATA_PATHS)
    x, y, w, timestamp = load_icenet_batch(data_path, with_dates=True)
    model = load_icenet_model(mask_out_land=cfg.mask_out_land_in_loss)

    outputs = np.zeros_like(y)
    grads = np.zeros_like(x)
    grads_std = np.zeros_like(x)

    ## Get explanations for several leadtimes
    # outputs = np.zeros((*y.shape, 6))
    # grads = np.zeros((*x.shape, 6))
    # grads_std = np.zeros((*x.shape, 6))
    # for leadtime in range(6):

    ## Get running mean of explanations for each month as new_outputs etc. (use running mean since limited memory)
    # leadtime_outputs = np.zeros_like(y)
    # leadtime_grads = np.zeros_like(x)
    # leadtime_grads_std = np.zeros_like(x)
    for month in range(12):
        print("-" * 80)
        print(f"Month: {month}")
        print("-" * 80)
        (
            month_outputs,
            month_grads,
            month_grads_std,
        ) = guided_backprop_dropout_ensemble_stats(
            x[month : month + 1],  # [0:1],
            y[month : month + 1],  # [0:1],
            w[month : month + 1],  # [0:1],
            model,
            n=cfg.n_dropout_variations,
            mask_out_land_in_loss=cfg.mask_out_land_in_loss,
            additional_mask=hudson_bay_mask,
            leadtime=leadtime,
        )
        ## accumulate running mean
        outputs = (month_outputs + month * outputs) / (month + 1)
        grads = (month_grads + month * grads) / (month + 1)
        grads_std = (month_grads_std + month * grads_std) / (month + 1)

    if cfg.mask_out_land_outputs:
        outputs = outputs * ~land_mask[:, :, None, None]
    if cfg.mask_out_land_grads:
        grads = grads * ~land_mask[:, :, None]
        grads_std = grads_std * ~land_mask[:, :, None]

    # outputs = leadtime_outputs
    # grads = leadtime_grads
    # grads_std = leadtime_grads_std

    # significant_mask = t_test(grads, grads_std, n=cfg.n_dropout_variations, alpha=0.05)
    return outputs, w, grads, grads_std, timestamp  # w[0:1, ..., cfg.leadtime]


def t_test(mu_hat, std_hat, n: int, alpha=0.05):
    """
    Use estimated mean and standard deviation to perform a
    t-test elementwise with a given alpha and number of samples. Return binary mask
    of significant features.

    Parameters
    ----------
        mu_hat, std_hat : np.array or similar
            Estimated mean and standard deviation of the feature importance
        n : int
            Number of samples used to estimate the mean and standard deviation.
        alpha : float
            Significance level of the t-test.

    Returns
    -------
        mask : np.array
            Binary mask of significant features. True if the feature is significant.
            The shape of the mask is the same as the shape of mu_hat and std_hat.

    Warning!: One sided test.
    """
    t_crit = stats.t.ppf(1 - alpha, n - 1)
    t = mu_hat / (std_hat / np.sqrt(n))
    return t > t_crit


def guided_backprop_dropout_ensemble_stats(
    x, y, w, model, n: int, mask_out_land_in_loss=True, additional_mask=None, leadtime=0
):
    """
    Perform guided backpropagation on a model with dropout.

    Args:
        x (np.array): Input data
        y (np.array): Target data
        w (np.array): A mask that indicates which pixels to include in the loss. This masks out land pixels, the polar hole
            and portions of the ocean that are assumed to never have ice. i.e. sort of irrelevant predictions.
        model (tf.keras.Model): The model to analyze.
        n (int): Number of dropout variations to perform.
        mask_out_land_in_loss (bool): Whether to mask out land pixels in the loss. If True, the loss will be zero for land pixels.
    """
    grad_l = []
    output_l = []

    for i in tqdm(range(n)):  ## TODO: Try sensitivity within
        # gb_model = model
        gb_model = Model(
            inputs=[model.inputs],
            outputs=[model.output],
        )
        with tf.compat.v1.get_default_graph().gradient_override_map(
            {"Relu": "GuidedRelu"}
        ):
            # layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]
            # for layer in layer_dict:
            #     if layer.activation == tf.keras.activations.relu:
            #         layer.activation = guidedRelu

            with tf.GradientTape() as tape:
                inputs = tf.cast(x, tf.float32)
                tape.watch(inputs)
                outputs = gb_model(inputs)

                ## sum over ice channels on first leadtime.
                # if mask_out_land_in_loss:
                #     outputs = outputs * ~np.load("land_mask.npy")[None, :, :, None, None]
                if mask_out_land_in_loss:
                    outputs = outputs * ~np.load(LAND_MASK_PATH)[None, :, :, None, None]
                    print("Masking out land")

                if additional_mask is not None:
                    outputs = outputs * additional_mask[None, :, :, None, None]
                    print("Masking out all outside hudson bay region.")

                warn("Masking out all inactive pixels in loss, not just land mask.")
                outputs_loss = outputs * tf.repeat(
                    w, 3, axis=3
                )  # repeat(w, "n h w 1 l -> n h w c l", c=3)
                loss = tf.reduce_mean(outputs_loss[0, :, :, 2:, leadtime])

            output_l.append(outputs)
            grads = tape.gradient(loss, inputs)[0]

            grads = np.array(grads)
            grad_l.append(grads)

    outputs = np.array(outputs).mean(axis=0)
    grad_l = np.array(grad_l)
    grads = grad_l.mean(axis=0)
    grads_std = grad_l.std(axis=0)
    return outputs, grads, grads_std


def plot_feature_importance(grads, grads_std, figname: str):
    """
    Plot feature importance with uncertainty. Not a very good plot, but is relatively fast and
    gives and idea.
    """
    f_means = grads.mean((0, 1))
    f_std = grads_std.mean((0, 1))
    f_lwr = f_means - 2 * f_std
    f_upr = f_means + 2 * f_std
    plt.plot(f_means)
    plt.fill_between(range(50), f_lwr, f_upr, alpha=0.6)
    plt.title("Feature importance")
    plt.savefig(figname)


def load_icenet_batch(path: str, with_dates=False):
    with np.load(path, allow_pickle=True) as data:
        x = data["x"]
        y = data["y"]
        w = data["w"]
        if with_dates:
            # timeframe = data["batch_IDs"]
            timestamp = data["timestamp"]
            return x, y, w, timestamp
        # print("only values of w", np.unique(w))
    return x, y, w


def load_icenet_model(mask_out_land=False):
    # model = unet_batchnorm_w_dropout_mod()
    # model.load_weights("network_dropout_mc_50.h5")

    custom_objects = {
        "categorical_focal_loss": construct_sum_last_two_channels_loss(
            mask_out_land=mask_out_land
        ),
        "ConstructLeadtimeAccuracy": ConstructLeadtimeAccuracy,
        "TemperatureScale": TemperatureScale,
        "DropoutWDefaultTraining": DropoutWDefaultTraining,
    }
    model = load_model(MODEL_PATH, custom_objects=custom_objects)

    return model


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


def test_load_icenet_batch():
    x, y, w, dates = load_icenet_batch(
        path="/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper/icenet/analyze_input_influences/test_data/test_data_batch_0_w_dates.npz",
        with_dates=True,
    )
    print(x.shape, y.shape)
    ## Inputs should be of dim: (batch_size, height, width, variable_channels)
    assert len(x.shape) == 4
    assert x.shape[1:] == (432, 432, 50)

    ## Labels should be of dim: (batch_size, height, width, sic_class, leadtime)
    assert len(y.shape) == 5
    assert y.shape[1:] == (432, 432, 3, 6)


##################################################################
##################################################################
##################################################################


@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_f = tf.cast(op.outputs[0] > 0, "float32")  # for f^l > 0
    gate_R = tf.cast(grad > 0, "float32")  # for R^l+1 > 0
    return gate_f * gate_R * grad

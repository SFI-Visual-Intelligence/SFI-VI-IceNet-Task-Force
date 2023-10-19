import tensorflow as tf
import numpy as np
from models import DropoutWDefaultTraining, TemperatureScale
from metrics import ConstructLeadtimeAccuracy


def load_icenet_monte_carlo_model(mask_out_land=False):
    """
    Load the IceNet model trained with MC dropout.
    """
    custom_objects = {
        "categorical_focal_loss": construct_sum_last_two_channels_loss(
            mask_out_land=mask_out_land
        ),
        "ConstructLeadtimeAccuracy": ConstructLeadtimeAccuracy,
        "TemperatureScale": TemperatureScale,
        "DropoutWDefaultTraining": DropoutWDefaultTraining,
    }

    model = tf.keras.models.load_model(
        "icenet/experimental/trained_models/network_dropout_mc_50.h5",
        custom_objects=custom_objects,
    )

    return model


def construct_sum_last_two_channels_loss(mask_out_land=False):
    """To be used with load_icenet_model. Not sure what this does."""
    def sum_last_two_channels_loss(y_true, y_pred, sample_weight=None):
        if mask_out_land != None:
            LAND_MASK = np.load("../data/masks/region_mask.npy") != 5
            y_pred = y_pred[:, ~LAND_MASK, ...]
        return tf.sum(y_pred[..., 1:2, :])

    return sum_last_two_channels_loss

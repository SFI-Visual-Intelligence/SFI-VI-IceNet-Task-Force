import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    UpSampling2D,
    concatenate,
    MaxPooling2D,
    Input,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from model_dependencies import (
    ConstructLeadtimeAccuracy,
    TemperatureScale,
    DropoutWDefaultTraining,
)


LAND_MASK = np.load("land_mask.npy")


def unet_batchnorm_w_dropout_mod(
    input_shape=(432, 432, 50),
    # loss,
    # weighted_metrics,
    # learning_rate=1e-4,
    filter_size=3,
    n_filters_factor=1,
    n_forecast_months=1,
    use_temp_scaling=True,
    n_output_classes=3,
    drop_out_rate=0.2,
    **kwargs
):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(
        np.int(64 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    conv1 = Conv2D(
        np.int(64 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(
        np.int(128 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    conv2 = Conv2D(
        np.int(128 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    conv3 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool3)
    ## -------------- Dropout layer added --------------
    conv4 = DropoutWDefaultTraining(drop_out_rate)(conv4)
    ## -------------------------------------------------
    conv4 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(
        np.int(512 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool4)
    ## -------------- Dropout layer added --------------
    conv5 = DropoutWDefaultTraining(drop_out_rate)(conv5)
    ## -------------------------------------------------
    conv5 = Conv2D(
        np.int(512 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    up6 = Conv2D(
        np.int(256 * n_filters_factor),
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2), interpolation="nearest")(bn5))
    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge6)
    ## -------------- Dropout layer added --------------
    conv6 = DropoutWDefaultTraining(drop_out_rate)(conv6)
    ## -------------------------------------------------
    conv6 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    up7 = Conv2D(
        np.int(256 * n_filters_factor),
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2), interpolation="nearest")(bn6))
    merge7 = concatenate([bn3, up7], axis=3)
    conv7 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge7)
    conv7 = Conv2D(
        np.int(256 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    up8 = Conv2D(
        np.int(128 * n_filters_factor),
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2), interpolation="nearest")(bn7))
    merge8 = concatenate([bn2, up8], axis=3)
    conv8 = Conv2D(
        np.int(128 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge8)
    conv8 = Conv2D(
        np.int(128 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    up9 = Conv2D(
        np.int(64 * n_filters_factor),
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2), interpolation="nearest")(bn8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        np.int(64 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge9)
    conv9 = Conv2D(
        np.int(64 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv9)
    conv9 = Conv2D(
        np.int(64 * n_filters_factor),
        filter_size,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv9)

    final_layer_logits = [
        (Conv2D(n_output_classes, 1, activation="linear")(conv9))
        for i in range(n_forecast_months)
    ]
    final_layer_logits = tf.stack(final_layer_logits, axis=-1)

    if use_temp_scaling:
        # Temperature scaling of the logits
        final_layer_logits_scaled = TemperatureScale()(final_layer_logits)
        final_layer = tf.nn.softmax(final_layer_logits_scaled, axis=-2)
    else:
        final_layer = tf.nn.softmax(final_layer_logits, axis=-2)

    model = Model(inputs, final_layer)

    # model.compile(
    #     optimizer=Adam(lr=learning_rate), loss=loss, weighted_metrics=weighted_metrics,
    # )

    return model


def sum_last_channel_loss(y_true, y_pred, sample_weight=None):
    # Clip the prediction value to prevent NaN's and Inf's
    # y_pred = y_pred[:, ~LAND_MASK, ...]
    return tf.sum(y_pred[..., 2, :])


def construct_sum_last_two_channels_loss(mask_out_land=False):
    # y_pred = y_pred[:, ~LAND_MASK, ...]
    def sum_last_two_channels_loss(y_true, y_pred, sample_weight=None):
        if mask_out_land:
            y_pred = y_pred[:, ~LAND_MASK, ...]
        return tf.sum(y_pred[..., 1:2, :])

    return sum_last_two_channels_loss


def construct_categorical_focal_loss(gamma=2.0):
    """
    Softmax version of focal loss.
      FL = - (1 - p_c)^gamma * log(p_c)
      where p_c = probability of correct class

    Parameters:
      gamma: Focusing parameter in modulating factor (1-p)^gamma
        (Default: 2.0, as mentioned in the paper)

    Returns:
    loss: Focal loss function for training.

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
    """

    def categorical_focal_loss(y_true, y_pred, sample_weight=None):
        """
        Parameters:
            y_true: Tensor of one-hot encoded true class values.
            y_pred: Softmax output of model corresponding to predicted
                class probabilities.

        Returns:
            focal_loss: Output tensor of pixelwise focal loss values.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss (downweights easy samples where the probability of
        #   correct class is high)
        focal_loss = K.pow(1 - y_pred, gamma) * cross_entropy

        # Loss is a tensor which is reduced implictly by TensorFlow using
        #   sample weights passed during training/evaluation
        return focal_loss

    return categorical_focal_loss

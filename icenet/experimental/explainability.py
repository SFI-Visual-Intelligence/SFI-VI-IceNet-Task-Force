"""
Tensorflow implementation of Guided Backpropagation. 
source: https://colab.research.google.com/drive/17tAC7xx2IJxjK700bdaLatTVeDA02GJn?usp=sharing#scrollTo=4-OduFD-wH14
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))  # if using jupyter kernel

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tqdm import tqdm
from einops import repeat
from warnings import warn
from models import unet_batchnorm_w_dropout
import experimental.config as config

# Directory of ensemble models
model_dir = "../trained_networks/unet_tempscale/networks/"

def compute_explanations(
    x, y, model, mask_out_land_inputs=False, mask_out_land_outputs=False, n=20
):
    land_mask = np.load("../data/masks/land_mask.npy")
    region_mask = np.load("../data/masks/region_mask.npy") != 5
    land_mask = region_mask
    hbs_mask = region_mask != 5
    if mask_out_land_inputs:
        x = x * land_mask[None, :, :, None]

    outputs, grads, grads_std = guided_backprop_dropout_ensemble_stats(x, y, model, n=n)

    ## Mask out land
    if mask_out_land_outputs:
        outputs = outputs * ~land_mask[:, :, None, None]
        grads = grads * ~land_mask[:, :, None]
        grads_std = grads_std * ~land_mask[:, :, None]

    return outputs, grads, grads_std


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
                    outputs = (
                        outputs
                        * ~np.load(config.LAND_MASK_PATH)[None, :, :, None, None]
                    )
                    print("Masking out land")

                if additional_mask is not None:
                    outputs = outputs * additional_mask[None, :, :, None, None]
                    print("Masking out all outside hudson bay region.")

                warn("Masking out all inactive pixels in loss, not just land mask.")
                outputs_loss = outputs * repeat(w, "n h w 1 l -> n h w c l", c=3)
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


def guided_backprop_dropout_ensemble(
    model, inputs, active_grid_cells, n: int, output_mask=None, leadtime=1
):
    """
    Perform guided backpropagation on a model with dropout.

    Args:
        model (tf.keras.Model): The model to analyze.
        inputs (np.array): Input data
        active_grid_cells (np.array): A mask that indicates which pixels to include in the output. This masks out land pixels, the polar hole
            and portions of the ocean that are assumed to never have ice. i.e. assumed irrelevant predictions.
        n (int): Number of dropout variations to perform.
        output_mask (np.array): A mask that indicates which pixels to include in the output.
        leadtime (int): The leadtime to analyze (1, ..., 6). Defaults to 1.

    Returns:
        outputs_list (list): A list of the model outputs for each dropout variation.
        grads_list (list): A list of the gradients for each dropout variation.
    """
    grads_list = []
    outputs_list = []

    for _ in tqdm(range(n)):
        gb_model = Model(
            inputs=[model.inputs],
            outputs=[model.output],
        )
        with tf.compat.v1.get_default_graph().gradient_override_map(
            {"Relu": "GuidedRelu"}
        ):
            with tf.GradientTape() as tape:
                inputs = tf.cast(inputs, tf.float32)
                tape.watch(inputs)
                outputs = gb_model(inputs)

                if output_mask is not None:
                    outputs = outputs * output_mask[None, :, :, None, None]
                    print("Masking out everything outside of output_mask.")

                warn("Masking out all inactive pixels in output.")
                outputs_loss = outputs * repeat(
                    active_grid_cells, "n h w 1 l -> n h w c l", c=3
                )
                loss = tf.reduce_mean(outputs_loss[0, :, :, 2:, leadtime - 1])

            outputs_list.append(outputs)
            grads = tape.gradient(loss, inputs)[0]

            grads = np.array(grads)
            grads_list.append(grads)

    return outputs_list, grads_list


def gradient_ensemble(
    model, inputs, active_grid_cells, output_mask=None, leadtime=1
):
    """
    Perform guided backpropagation on a model with dropout.

    Args:
        model (tf.keras.Model): The model to analyze.
        inputs (np.array): Input data
        active_grid_cells (np.array): A mask that indicates which pixels to include in the output. This masks out land pixels, the polar hole
            and portions of the ocean that are assumed to never have ice. i.e. assumed irrelevant predictions.
        output_mask (np.array): A mask that indicates which pixels to include in the output.
        leadtime (int): The leadtime to analyze (1, ..., 6). Defaults to 1.

    Returns:
        outputs_list (list): A list of the model outputs for each dropout variation.
        grads_list (list): A list of the gradients for each dropout variation.
    """
    grads_list = []
    outputs_list = []

    #for _ in tqdm(range(n)):
    for model_path in tqdm(os.listdir(model_dir)):
        #model = Model(
        #    inputs=[model.inputs],
        #    outputs=[model.output],
        #)
        model = tf.keras.models.load_model(os.path.join(model_dir, model_path))
        with tf.GradientTape() as tape:
            inputs = tf.cast(inputs, tf.float32)
            tape.watch(inputs)
            outputs = model(inputs)
            if output_mask is not None:
                outputs = outputs * output_mask[None, :, :, None, None]
                
            outputs_loss = outputs * repeat(
                active_grid_cells, "n h w 1 l -> n h w c l", c=3
            )
            loss = tf.reduce_mean(outputs_loss[0, :, :, 2:, leadtime - 1])

        outputs_list.append(outputs_loss)
        grads = tape.gradient(loss, inputs)[0]

        grads = np.array(grads)
        grads_list.append(grads)

    return outputs_list, grads_list


def integrated_gradient_dropout_ensemble(
    model, inputs, active_grid_cells, n: int, output_mask=None, leadtime=1
):
    """
    Do integrated gradients on a model with dropout.
    """
    grads_list = []
    outputs_list = []
    inputs = inputs[0]
    # define baseline
    baseline = np.zeros_like(inputs)
    # iterate over dropout variations
    for _ in tqdm(range(n)):
        outputs, grads = integrated_gradients(
            model, baseline, inputs, active_grid_cells, output_mask, leadtime, m_steps=3
        )
        outputs_list.append(outputs)
        # grads = tape.gradient(outputs, inputs)[0]
        grads = np.array(grads)
        grads_list.append(grads)

    return outputs_list, grads_list


@tf.function
def integrated_gradients(
    model, baseline, image, active_grid_cells, output_mask, leadtime, m_steps=50
):
    """
    Compute integrated gradients for a model.
    integrated_gradients.shape = image.shape
    """
    # Generate sequence of alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    # Generate interpolated images between baseline and input image.
    interpolated_path_input_batch = generate_path_inputs(
        baseline=baseline, image=image, alphas=alphas
    )
    # Compute gradients for model output wrt batch of interpolated images.
    _, gradient_batch = compute_gradients(
        model, interpolated_path_input_batch, active_grid_cells, output_mask, leadtime
    )
    # Integral approximation through averaging gradients for each feature map in the last axis.
    avg_gradients = []
    for i in range(gradient_batch.shape[-1]):
        avg_gradients.append(
            integral_approximation(gradients=gradient_batch[:, :, :, i])
        )
    avg_gradients = tf.stack(avg_gradients, axis=-1)
    # Scale integrated gradients with respect to input.
    integrated_gradients = (
        tf.cast(image - baseline, avg_gradients.dtype) * avg_gradients
    )

    # return _, tf.reduce_sum(integrated_gradients, axis=2)
    return _, integrated_gradients


def compute_gradients(model, inputs, active_grid_cells, output_mask=None, leadtime=1):
    """
    Compute gradients.
    grads.shape is [batch_size, height, width, channels].
    """
    with tf.GradientTape() as tape:
        inputs = tf.cast(inputs, tf.float32)
        tape.watch(inputs)
        outputs = model(inputs)
        # mask out everything outside of the output mask
        if output_mask is not None:
            outputs = outputs * tf.cast(output_mask[None, :, :, None, None], tf.float32)
        # mask out everything outside of the active grid cells
        outputs = outputs * repeat(active_grid_cells, "n h w 1 l -> n h w c l", c=3)
        # take the mean of the output, only look at >80% SIC predictions
        outputs = tf.reduce_mean(outputs[:, :, :, 2:, leadtime - 1])
    grads = tape.gradient(outputs, inputs)
    return outputs, grads


def generate_path_inputs(baseline, image, alphas):
    """
    Generates m interpolated images between input and baseline image.
    Parameters
    ----------
    baseline_img : numpy.ndarray
        3D tensor of floats.
    input_img : numpy.ndarray
        3D tensor of floats.
    alphas : numpy.ndarray
        Sequence of alpha values.
    Returns path_inputs
    -------
    4D tf.tensor of step images.
    """
    if not len(baseline.shape) == len(image.shape) == 3:
        raise Exception("Input images must have shape (W, H, C)")

    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.cast(tf.expand_dims(baseline, 0), tf.float32)
    input_x = tf.cast(tf.expand_dims(image, 0), tf.float32)

    delta = input_x - baseline_x
    path_inputs = baseline_x + alphas_x * delta

    return tf.convert_to_tensor(path_inputs)


def integral_approximation(gradients):
    """
    Approximate integration of input using Riemann sums
    and the trapezoidal rule.
    Parameters
    ----------
    gradients : tf.Tensor
        Can have any shape.
    Returns
    -------
    integrated_gradients : tf.Tensor
        Shape as input.
    """
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0, dtype=gradients.dtype)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def plot_feature_importance(grads, grads_std, figname: str):
    f_means = grads.mean((0, 1))
    f_std = grads_std.mean((0, 1))
    f_lwr = f_means - 2 * f_std
    f_upr = f_means + 2 * f_std
    plt.plot(f_means)
    plt.fill_between(range(50), f_lwr, f_upr, alpha=0.6)
    plt.title("Feature importance")
    plt.savefig(figname)


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad

##################################################################
##################################################################
##################################################################


@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_f = tf.cast(op.outputs[0] > 0, "float32")  # for f^l > 0
    gate_R = tf.cast(grad > 0, "float32")  # for R^l+1 > 0
    return gate_f * gate_R * grad


if __name__ == "__main__":
    x = None
    
    y = None

    model = unet_batchnorm_w_dropout()
    with tf.compat.v1.get_default_graph().gradient_override_map({"Relu": "GuidedRelu"}):
        gb_model = model(
            inputs=[model.inputs], outputs=[model.get_layer("conv5_block3_out").output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(x, tf.float32)
            tape.watch(inputs)
            outputs = gb_model(inputs)

        grads = tape.gradient(outputs, inputs)[0]

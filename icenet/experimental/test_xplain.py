import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))  # if using jupyter kernel

import numpy as np
import tensorflow as tf
from utils import IceNetDataLoader
import config
from experimental.guided_backprop import (
    generate_path_inputs,
    compute_gradients,
    integrated_gradients,
    integral_approximation,
    integrated_gradient_dropout_ensemble,
)
from experimental.utils import load_icenet_model
from experimental.config import LAND_MASK_PATH

# Instantiate dataloader
dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)
dataloader = IceNetDataLoader(dataloader_config_fpath)

# Generate data for testing
start_date = "2012-01-01"
inputs, _, active_grid_cells = dataloader.data_generation(start_date)
output_mask = ~np.load(LAND_MASK_PATH)
baseline = np.zeros_like(inputs[0])

# Load model
model = load_icenet_model()


def test_integrated_gradients():
    m_steps = 10
    _, gradients = integrated_gradients(
        model, baseline, inputs[0], active_grid_cells, output_mask, 1, m_steps
    )
    assert gradients.shape == inputs[0].shape


def test_compute_gradients():
    m_steps = 10
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    interpolated_path_input_batch = generate_path_inputs(baseline, inputs[0], alphas)
    _, gradients = compute_gradients(
        model, interpolated_path_input_batch, active_grid_cells, output_mask, 1
    )
    assert gradients.shape == interpolated_path_input_batch.shape


def test_generate_path_inputs():
    m_steps = 10
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    interpolated_path_input_batch = generate_path_inputs(baseline, inputs[0], alphas)
    assert interpolated_path_input_batch.shape == (m_steps + 1, *inputs[0].shape)

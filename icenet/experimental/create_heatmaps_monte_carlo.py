import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))  # if using jupyter kernel

import time
import numpy as np
import pandas as pd
import tensorflow as tf

from einops import repeat
from tqdm import tqdm

import config
import utils

from experimental.utils import load_icenet_monte_carlo_model

####################################################################
### USER INPUT #####################################################
####################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--output_mask", type=str, default='', help="path/to/mask.npy")
parser.add_argument("--target_date", type=str, default="2013-09-01", help="YYYY-MM-DD")
parser.add_argument("--n_forecast_months", type=int, default=6)
parser.add_argument("--dataloader_ID", type=str, default="2021_06_15_1854_icenet_nature_communications")
parser.add_argument("--results_fpath", type=str, default="icenet/experimental/results")
parser.add_argument("--num_models", type=int, default=50)
args = parser.parse_args()

if __name__ == "__main__":

    ####################################################################
    ### PREPARE INPUTS #################################################
    ####################################################################

    try:
        print("Loading mask: ", args.output_mask)
        output_mask = np.load(args.output_mask)
    except FileNotFoundError:
        print("No mask provided.")
        output_mask = np.ones((432, 432), dtype=bool)

    # Instantiate dataloader
    dataloader_config_fpath = os.path.join(config.dataloader_config_folder, args.dataloader_ID + ".json")
    dataloader = utils.IceNetDataLoader(dataloader_config_fpath)

    # List variable names in order of appearance as in Figure 7
    all_ordered_variable_names = dataloader.determine_variable_names()
    leadtimes = np.arange(1, args.n_forecast_months + 1)

    # Build init dates
    target_dates = pd.DatetimeIndex([args.target_date])
    init_dates = pd.DatetimeIndex([])
    for target_date in target_dates:
        init_dates = init_dates.append(pd.date_range(start=target_date - pd.DateOffset(months=6-1), end=target_date, freq="MS"))

    ## Load model
    model = load_icenet_monte_carlo_model(mask_out_land=False)

    # Load inputs
    print("Building up all the baseline inputs... ")
    inputs_list = []
    active_grid_cells_list = []
    for forecast_start_date in init_dates:
        inputs, _, active_grid_cells = dataloader.data_generation(forecast_start_date)
        inputs_list.append(inputs[0])
        active_grid_cells_list.append(active_grid_cells)
    inputs = np.stack(inputs_list, axis=0)

    ####################################################################
    ### TEST BLOCK #####################################################
    ####################################################################

    for dat_i, target_date in enumerate(target_dates):
        for leadtime in leadtimes:
            forecast_start_date = init_dates[dat_i + 6 - leadtime]
            assert forecast_start_date  + pd.DateOffset(months=leadtime-1) == target_date

    ####################################################################
    ### MAIN ###########################################################
    ####################################################################
    
    # Create arrays to store results
    heatmap = np.zeros((50, 432, 432, 6))
    forecasts = np.zeros((432, 432, 6, 3))
    
    print("Producing gradients...")
    # Iterate over all models and forecast dates
    for leadtime in leadtimes:        
        for _ in tqdm(range(args.num_models), total=args.num_models):
            with tf.GradientTape() as tape:
                active_grid_cells = active_grid_cells_list[6 - leadtime]
                inputs = inputs_list[6 - leadtime][None, ...]
                inputs = tf.cast(inputs, tf.float32)
                tape.watch(inputs)
                outputs = model(inputs)
                outputs = outputs * output_mask[None, :, :, None, None]
                outputs_loss = outputs * repeat(active_grid_cells, "n h w 1 l -> n h w c l", c=3)
                loss = tf.reduce_mean(outputs_loss[0, :, :, 2:, leadtime - 1])
            gradients = tape.gradient(loss, inputs)[0]
            gradients = np.array(gradients)

            heatmap[:, :, :, leadtime - 1] += np.moveaxis(gradients, -1, 0) / args.num_models
            forecasts[:, :, leadtime - 1, :] += outputs[0, :, :, :, leadtime - 1] / args.num_models
        
    # Save results
    os.makedirs(args.results_fpath, exist_ok=True)
    timestr = time.strftime("%d%m%y_%H:%M")

    try:
        np.savez(os.path.join(args.results_fpath, "spatial_heatmap_" + timestr + ".npz"), heatmap)
        np.savez(os.path.join(args.results_fpath, "spatial_forecasts_" + timestr + ".npz"), forecasts)
    except OSError:
        print(
            "Could not save results dataframe to provided path. Saving to current directory instead."
        )
        np.savez(os.path.join(os.getcwd(), "spatial_heatmap_" + timestr + ".npz"), heatmap)
        np.savez(os.path.join(os.getcwd(), "spatial_forecasts_" + timestr + ".npz"), forecasts)

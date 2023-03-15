import sys
import os

os.environ["OMP_NUM_THREADS"] = "16"

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))  # if using jupyter kernel
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
import utils
import time
import argparse
from experimental.guided_backprop import guided_backprop_dropout_ensemble, gradient_dropout_ensemble
from experimental.utils import load_icenet_model
from experimental.config import REGION_MASK_PATH, LAND_MASK_PATH

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
pd.options.display.float_format = "{:.2f}".format


########
####################################################################
### USER INPUT ######################################a###############
####################################################################
########

parser = argparse.ArgumentParser()
parser.add_argument("--mask", type=str, default="None")
args = parser.parse_args()

region_mask = np.load(REGION_MASK_PATH)
land_mask = ~np.load(LAND_MASK_PATH)
hbs_mask = region_mask == 5
bering_mask = region_mask == 4

if args.mask == "hbs":
    output_mask = hbs_mask
elif args.mask == "land":
    output_mask = land_mask
elif args.mask == "bering":
    output_mask = bering_mask
else:
    output_mask = None

print("Mask: ", args.mask)

dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
start_date = "2012-01-01"
end_date = "2019-12-01"
n_forecast_months = 6
dropout_sample_size = 25
results_df_fpath = "icenet/experimental/results"

########
####################################################################
### PREPARE INPUTS #################################################
####################################################################
########

# Instantiate dataloader
dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)
dataloader = utils.IceNetDataLoader(dataloader_config_fpath)

# List variable names in order of appearance as in Figure 7
all_ordered_variable_names = dataloader.determine_variable_names()
leadtimes = np.arange(1, n_forecast_months + 1)
model_numbers = ["model" for _ in range(dropout_sample_size)]

# All dates for which we want to make a forecast
target_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

# Forecast initialisation dates s.t. each target date has a forecast at each lead time
init_dates = pd.date_range(
    start=pd.Timestamp(start_date) - pd.DateOffset(months=n_forecast_months - 1),
    end=end_date,
    freq="MS",
)

# Load model
model = load_icenet_model(False)

# Load inputs
print("Building up all the baseline inputs... ")
inputs_list = []
active_grid_cells_list = []
for forecast_start_date in init_dates:
    inputs, _, active_grid_cells = dataloader.data_generation(forecast_start_date)
    inputs_list.append(inputs[0])
    active_grid_cells_list.append(active_grid_cells)
inputs = np.stack(inputs_list, axis=0)
print("Done.\n\n")

# Create heatmap
heatmap = np.zeros((50, 432, 432, 6))

########
####################################################################
### MAIN ###########################################################
####################################################################
########


# Iterate over all models and forecast dates
for dat_i, target_date in tqdm(enumerate(target_dates), total=len(target_dates)):
    for leadtime in leadtimes:
        inputs = inputs_list[dat_i + 6 - leadtime][None, ...]
        active_grid_cells = active_grid_cells_list[dat_i + 6 - leadtime]
        ###
        ###
        ### Insert feature importance code here
        ###
        ###
        _, feature_importance_list = gradient_dropout_ensemble(
            model=model,
            inputs=inputs,
            active_grid_cells=active_grid_cells,
            n=dropout_sample_size,
            output_mask=output_mask,
            leadtime=leadtime,
        )
        ###
        ###
        ### End of feature importance code
        ###
        ###
        # average over all models
        feature_importance = np.mean(feature_importance_list, axis=0)
        # add to heatmap
        heatmap[:, :, :, leadtime - 1] += feature_importance
        
    print("\nDone.\n")

# Save results
os.makedirs(os.path.dirname(results_df_fpath), exist_ok=True)
timestr = time.strftime("%d%m%y_%H:%M")
filename = "spatial_heatmap_" + timestr + ".csv"

try:
    np.savez(os.path.join(results_df_fpath, "spatial_heatmap_" + timestr + ".npz"), heatmap)
except OSError:
    print(
        "Could not save results dataframe to provided path. Saving to current directory instead."
    )
    np.savez(os.path.join(os.getcwd(), "spatial_heatmap_" + timestr + ".npz"), heatmap)

textfilename = os.path.join(results_df_fpath, "spatial_heatmap_" + timestr + ".txt")

with open(textfilename, "w") as text_file:
    text_file.write("Mask: %s" % args.mask)
    text_file.write("Dropout sample size: %s" % dropout_sample_size)

print("Done")

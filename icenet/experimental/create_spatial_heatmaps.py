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
extreme_mask = np.load('icenet/experimental/figures/2013_extreme_mask.npy')


if args.mask == "hbs":
    output_mask = hbs_mask
elif args.mask == "land":
    output_mask = land_mask
elif args.mask == "bering":
    output_mask = bering_mask
elif args.mask == "extreme":
    output_mask = extreme_mask
else:
    output_mask = None

print("Mask: ", args.mask)

dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
start_date = "2012-01-01"
end_date = "2019-12-01"
n_forecast_months = 6
dropout_sample_size = 25
results_fpath = "icenet/experimental/results"

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

# Uncomment to only use a specific subset of dates
target_dates = pd.DatetimeIndex(["2014-09-01"])
init_dates = pd.DatetimeIndex([])
for target_date in target_dates:
    init_dates = init_dates.append(pd.date_range(start=target_date - pd.DateOffset(months=6-1), end=target_date, freq="MS"))


####################################################################
### TEST BLOCK #####################################################
####################################################################

#for dat_i, target_date in enumerate(target_dates):
#    for leadtime in leadtimes:
#        forecast_start_date = init_dates[dat_i + 6 - leadtime]
#        assert forecast_start_date  + pd.DateOffset(months=leadtime-1) == target_date

####################################################################
####################################################################
####################################################################


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

# Create arrays to store results
heatmap = np.zeros((50, 432, 432, 6))
forecasts = np.zeros((432, 432, 6, 3))

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
        output_list, feature_importance_list = gradient_dropout_ensemble(
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
        print(np.mean(output_list, axis=0).shape)
        forecast = np.mean(output_list, axis=0)[0, :, :, :, leadtime - 1]
        # add to heatmap
        heatmap[:, :, :, leadtime - 1] += np.moveaxis(feature_importance, -1, 0)
        forecasts[:, :, leadtime - 1, :] += forecast
        
    print("\nDone.\n")

# Save results
os.makedirs(os.path.dirname(results_fpath), exist_ok=True)
timestr = time.strftime("%d%m%y_%H:%M")

try:
    np.savez(os.path.join(results_fpath, "spatial_heatmap_" + timestr + ".npz"), heatmap)
    np.savez(os.path.join(results_fpath, "spatial_forecasts_" + timestr + ".npz"), forecasts)
except OSError:
    print(
        "Could not save results dataframe to provided path. Saving to current directory instead."
    )
    np.savez(os.path.join(os.getcwd(), "spatial_heatmap_" + timestr + ".npz"), heatmap)
    np.savez(os.path.join(os.getcwd(), "spatial_forecasts_" + timestr + ".npz"), forecasts)

textfilename = os.path.join(results_fpath, "spatial_heatmap_" + timestr + ".txt")

with open(textfilename, "w") as text_file:
    text_file.write("Mask: %s\n" % args.mask)
    text_file.write("Dropout sample size: %s\n" % dropout_sample_size)
    # write  all target dates one line per date
    text_file.write("Target dates: %s" % target_dates.strftime("%Y-%m-%d").to_list())

print("Done")

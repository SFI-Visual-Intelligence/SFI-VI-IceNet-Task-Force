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
from experimental.explainability import guided_backprop_dropout_ensemble, gradient_dropout_ensemble, integrated_gradient_dropout_ensemble
from experimental.utils import load_icenet_model
from experimental.config import REGION_MASK_PATH, LAND_MASK_PATH

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
pd.options.display.float_format = "{:.2f}".format


def get_aggregation_function(agg_func_name):
    if agg_func_name == "mean":
        return np.mean
    elif agg_func_name == "sum":
        return np.sum
    elif agg_func_name == "abssum":
        return lambda x: np.sum(np.abs(x))
    elif agg_func_name == "max":
        return np.max
    elif agg_func_name == "absmax":
        return lambda x: np.max(np.abs(x))
    else:
        raise ValueError(f"Aggregation function {agg_func_name} not supported.")


########
####################################################################
### USER INPUT ######################################a###############
####################################################################
########

parser = argparse.ArgumentParser()
parser.add_argument("--mask", type=str, default="None")
parser.add_argument("--agg_func", type=str, default="sum")
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

aggregate_features = get_aggregation_function(args.agg_func)

print("Aggregation function: ", args.agg_func)

dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
start_date = "2012-01-01"
end_date = "2019-12-01"
n_forecast_months = 6
dropout_sample_size = 25
results_df_fpath = "icenet/experimental/results"

########
####################################################################
### MAIN ###########################################################
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
model_numbers = ["model" + str(i) for i in range(dropout_sample_size)]

# All dates for which we want to make a forecast
target_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

# Forecast initialisation dates s.t. each target date has a forecast at each lead time
init_dates = pd.date_range(
    start=pd.Timestamp(start_date) - pd.DateOffset(months=n_forecast_months - 1),
    end=end_date,
    freq="MS",
)

# Create DataFrame for storing results
multi_index = pd.MultiIndex.from_product(
    [model_numbers, leadtimes, target_dates, all_ordered_variable_names],
    names=["Model number", "Leadtime", "Forecast date", "Variable"],
)

results_df = pd.DataFrame(
    index=multi_index, columns=["Feature importance"], dtype=np.float32
).sort_index()

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


# Iterate over all models and forecast dates
for dat_i, target_date in tqdm(enumerate(target_dates), total=len(target_dates)):
    print(f"Forecasting for {target_date}...")
    for leadtime in leadtimes:
        print(f"Leadtime {leadtime}...")
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
        for var_i, varname in enumerate(tqdm(all_ordered_variable_names, leave=False)):
            for model_number in tqdm(range(dropout_sample_size), leave=False):
                feature_importance = feature_importance_list[model_number][..., var_i]
                aggregated_feature_importance = aggregate_features(feature_importance)
                idx = (model_numbers[model_number], leadtime, target_date, varname)
                results_df.loc[idx, "Feature importance"] = aggregated_feature_importance
                #results_df.loc[
                #    model_number, leadtime, target_date, varname
                #] = aggregated_feature_importance
        
    print("Done with forecast date ", target_date)

# Save results
os.makedirs(os.path.dirname(results_df_fpath), exist_ok=True)
timestr = time.strftime("%d%m%y_%H:%M")
filename = "feature_importance_" + timestr + ".csv"

try:
    results_df.to_csv(os.path.join(results_df_fpath, filename))
except OSError:
    print(
        "Could not save results dataframe to provided path. Saving to current directory instead."
    )
    results_df.to_csv(os.path.join(os.getcwd(), filename))

textfilename = os.path.join(results_df_fpath, "feature_importance_" + timestr + ".txt")

with open(textfilename, "w") as text_file:
    text_file.write("Mask: %s" % args.mask)
    text_file.write("\nAggregation function: %s" % args.agg_func)
    text_file.write("Dropout sample size: %s" % dropout_sample_size)

print("Done")

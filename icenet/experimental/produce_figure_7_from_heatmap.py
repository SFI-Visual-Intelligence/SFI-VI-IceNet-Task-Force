import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))

import config
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

results_folder = "icenet/experimental/results"
filename_1 = "spatial_heatmap_190423_09:17.npz" # september 2013 extreme event
filename = "spatial_heatmap_190423_14:14.npz" # september 2007 no sea ice
filename = "spatial_heatmap_240423_13:19.npz" # june 1999 hbs no sea ice
filename = "spatial_heatmap_240423_13:24.npz" # june 2009 hbs sic
filename = "spatial_heatmap_260423_13:33.npz" # september 2014 extreme mask
aggregate_features = lambda x: np.sum(x) # can use another aggregation function here
figure_destionation_folder = "icenet/experimental/figures"

### Load heatmap
####################################################################

heatmap = np.load(os.path.join(results_folder, filename))["arr_0"]

### List variable names as they appear in figure 7, supplementary material
####################################################################

dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)
dataloader = utils.IceNetDataLoader(dataloader_config_fpath)
all_ordered_variable_names = dataloader.determine_variable_names()


### Create dataframe from heatmap
####################################################################

model_numbers = ["Model"] # just one model
leadtimes = np.arange(1, 7)
target_dates = pd.DatetimeIndex(["2001-01-01"]) # just a random date

multi_index = pd.MultiIndex.from_product(
    [all_ordered_variable_names, model_numbers, leadtimes, target_dates],
    names=["Model number", "Leadtime", "Forecast date", "Variable"],
)
results_df = pd.DataFrame(
    index=multi_index, columns=["Feature importance"], dtype=np.float32
)

# Save heatmap to dataframe
for leadtime in leadtimes:
    for var_i, varname in enumerate(tqdm(all_ordered_variable_names, leave=False)):
        feature_importance = heatmap[var_i, :, :, leadtime-1]
        aggregated_feature_importance = aggregate_features(feature_importance)
        # only one model and one target date
        results_df.loc[0, leadtime, target_dates[0], varname] = aggregated_feature_importance

### Produce Figure 7
####################################################################

# Compute mean over all models and forecast dates
mean_results_df = results_df.groupby(["Leadtime", "Variable"]).mean()

# Reorder rows and columns
mean_results_heatmap = (
    mean_results_df.reset_index()
    .pivot(index="Variable", columns="Leadtime")
    .reindex(all_ordered_variable_names)["Feature importance"]
)


#
#
# Do the same for the other heatmap
#
#heatmap_1 = np.load(os.path.join(results_folder, filename_1))["arr_0"]
#results_df_1 = pd.DataFrame(
#    index=multi_index, columns=["Feature importance"], dtype=np.float32
#)
#for leadtime in leadtimes:
#    for var_i, varname in enumerate(tqdm(all_ordered_variable_names, leave=False)):
#        feature_importance = heatmap_1[var_i, :, :, leadtime-1]
#        aggregated_feature_importance = aggregate_features(feature_importance)
#        # only one model and one target date
#        results_df_1.loc[0, leadtime, target_dates[0], varname] = aggregated_feature_importance
#mean_results_df_1 = results_df_1.groupby(["Leadtime", "Variable"]).mean()
#mean_results_heatmap_1 = (
#    mean_results_df_1.reset_index()
#    .pivot(index="Variable", columns="Leadtime")
#    .reindex(all_ordered_variable_names)["Feature importance"]
#)
#
#
#
#

#mean_results_heatmap = mean_results_heatmap * (mean_results_heatmap * mean_results_heatmap_1 < 0)

### Zero out elements where the sign of the two heatmaps is the same
#mean_results_df * (np.sign(mean_results_df) != np.sign(mean_results_df_1))



# To zero out uninteresting features, uncomment the following lines
#mean_results_heatmap.iloc[:18, :] = 0
#mean_results_heatmap.iloc[-3:, :] = 0

# To scale heatmap values, uncomment the following lines
#mean_results_heatmap -= np.min(mean_results_heatmap.values[:-3], axis=0)
#mean_results_heatmap /= np.max(mean_results_heatmap.values[:, :-1])
mean_results_heatmap /= np.max(mean_results_heatmap.values, axis=0)
#mean_results_heatmap *= 11

# Reset index to make variable names appear in the heatmap
mean_results_df = mean_results_df.reset_index()

cbar_kws = {}
cbar_kws["label"] = "Feature importance"

verbose_varnames = []
short_varnames = mean_results_heatmap.index.values.astype("str")
for varname in short_varnames:
    verbose_varname = utils.make_varname_verbose_any_leadtime(varname)
    verbose_varnames.append(verbose_varname)
mean_results_heatmap.index = verbose_varnames

with plt.rc_context(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.labelsize": 9,
    }
):
    fig, ax = plt.subplots(figsize=(6, 9))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)

    sns.heatmap(
        data=mean_results_heatmap,
        annot=True,
        annot_kws=dict(fontsize=7),
        fmt=".2f",
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws=cbar_kws,
        vmax=1.0,
        vmin=-1.0,
        cbar_ax=cax,
    )
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("Input variable name")
    cax.set_frame_on(True)
    ax.set_frame_on(True)
    plt.tight_layout()

os.makedirs(figure_destionation_folder, exist_ok=True)

plt.savefig(os.path.join(figure_destionation_folder, "supp_fig7_" + filename.split(".")[0] + ".png"))
# uncomment the following line to save as pdf
#plt.savefig(os.path.join(figure_destionation_folder, "supp_fig7_" + filename + ".pdf"))
plt.close()

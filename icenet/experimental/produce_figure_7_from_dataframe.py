"""
To be used in conjunction with the pandas dataframes produced by icenet/experimental/create_feature_importance_dataframe.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))
import config
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


### User input
####################################################################

dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
figure_destionation_folder = "icenet/experimental/figures"
results_folder = "icenet/experimental/results"
filename = "landmask_sum_guided_backprop.csv"

### Load results dataframe
####################################################################

dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)
dataloader = utils.IceNetDataLoader(dataloader_config_fpath)
results_df = pd.read_csv(os.path.join(results_folder, filename))

# To only look at June forecasts, uncomment the following lines
#test = results_df["Forecast date"].str.contains("06-01") + results_df["Variable"].str.contains("06-01")
#results_df = results_df[test]

### Produce Figure 7
####################################################################

# List variable names as they appear in figure 7, supplementary material
all_ordered_variable_names = dataloader.determine_variable_names()

# Compute mean over all models and forecast dates
mean_results_df = results_df.groupby(["Leadtime", "Variable"]).mean()

# Reorder rows and columns
mean_results_heatmap = (
    mean_results_df.reset_index()
    .pivot(index="Variable", columns="Leadtime")
    .reindex(all_ordered_variable_names)["Feature importance"]
)

# To zero out uninteresting features, uncomment the following lines
#mean_results_heatmap.iloc[:18, :] = 0
#mean_results_heatmap.iloc[-3:, :] = 0

# Scale heatmap values
#mean_results_heatmap -= np.min(mean_results_heatmap.values, axis=0)
#mean_results_heatmap /= np.max(mean_results_heatmap.values[:, :-1])
#mean_results_heatmap *= 11
from experimental.config import LAND_MASK_PATH, REGION_MASK_PATH
n = np.sum(~np.load(LAND_MASK_PATH))
#n = np.sum(np.load(REGION_MASK_PATH) == 5)
mean_results_heatmap /= (n*5e-7)


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
        vmax=.5,
        vmin=-.5,
        cbar_ax=cax,
    )
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("Input variable name")
    cax.set_frame_on(True)
    ax.set_frame_on(True)
    plt.tight_layout()

os.makedirs(figure_destionation_folder, exist_ok=True)

plt.savefig(os.path.join(figure_destionation_folder, "supp_fig7_" + filename + ".png"))
plt.savefig(os.path.join(figure_destionation_folder, "supp_fig7_" + filename + ".pdf"))
plt.close()

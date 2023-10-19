import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))

import config
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser()
parser.add_argument("--heatmap", type=str, default="None", help="path/to/heatmap.npy")
parser.add_argument("--destination_folder", type=str, default="icenet/experimental/figures", help="path/to/fig")
parser.add_argument("--dataloader_ID", type=str, default="2021_06_15_1854_icenet_nature_communications")
args = parser.parse_args()

if __name__ == "__main__":

    # =============================================================================
    # === PREPARE INPUTS ==========================================================
    # =============================================================================
    
    ### Load heatmap and mask out land
    land_mask = ~np.load("./icenet/experimental/masks/land_mask.npy")
    heatmap = np.load(args.heatmap)["arr_0"]
    heatmap = heatmap * land_mask[None, :, :, None]
    
    ### List variable names as they appear in figure 7, supplementary material
    dataloader_config_fpath = os.path.join(config.dataloader_config_folder, args.dataloader_ID + ".json")
    dataloader = utils.IceNetDataLoader(dataloader_config_fpath)
    all_ordered_variable_names = dataloader.determine_variable_names()

    ### Create dataframe from heatmap
    leadtimes = np.arange(1, 7)

    multi_index = pd.MultiIndex.from_product(
        [all_ordered_variable_names, leadtimes],
        names=["Variable", "Leadtime"],
    )

    results_df = pd.DataFrame(
        index=multi_index, columns=["Feature importance"], dtype=np.float32
    )

    ## transfer heatmap values to dataframe
    for leadtime in leadtimes:
        for var_i, varname in enumerate(tqdm(all_ordered_variable_names, leave=False)):
            feature_importance = heatmap[var_i, :, :, leadtime-1]
            aggregated_feature_importance = np.mean(feature_importance)
            results_df.loc[varname, leadtime] = aggregated_feature_importance

    # =============================================================================
    # === PLOT HEATMAP ============================================================
    # =============================================================================

    ## Reorder rows and columns
    mean_results_heatmap = (
        results_df.reset_index()
        .pivot(index="Variable", columns="Leadtime")
        .reindex(all_ordered_variable_names)["Feature importance"]
    )
    
    ## Normalize
    mean_results_heatmap /= np.max(mean_results_heatmap.values)

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
    
    ## Save figure
    os.makedirs(args.destination_folder, exist_ok=True)

    filename = os.path.basename(args.heatmap).split(".")[0]

    plt.savefig(os.path.join(args.destination_folder, filename + ".pdf"))

    plt.close()

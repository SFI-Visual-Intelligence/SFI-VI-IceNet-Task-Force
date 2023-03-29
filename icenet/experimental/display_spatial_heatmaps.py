import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import config
from icenet.experimental.config import LAND_MASK_PATH
from icenet.experimental.make_hbs_map import createLandEdgeImage


land_mask = np.load(LAND_MASK_PATH)
land_edge = createLandEdgeImage()


########################################################################################
### load data ##########################################################################
########################################################################################
path_to_file = "./icenet/experimental/results/spatial_heatmap_150323_16:32.npz" # average over all predictions
path_to_file = "./icenet/experimental/results/spatial_heatmap_220323_14:47.npz" # september 2012 prediction for HBS
path_to_file = "./icenet/experimental/results/spatial_heatmap_220323_16:13.npz" # september 2012 prediction global
path_to_file = "./icenet/experimental/results/spatial_heatmap_290323_16:20.npz" # december 2012 prediction HBS
path_to_forecast = "./icenet/experimental/results/spatial_forecasts_290323_16:20.npz" # december 2012 forecast HBS
img = np.load(path_to_file)["arr_0"]
forecast = np.load(path_to_forecast)["arr_0"]
########################################################################################
########################################################################################
########################################################################################


########################################################################################
### load variable names ################################################################
########################################################################################
dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)
dataloader = utils.IceNetDataLoader(dataloader_config_fpath)
all_ordered_variable_names = dataloader.determine_variable_names()

for i, varname in enumerate(all_ordered_variable_names):
    all_ordered_variable_names[i] = utils.make_varname_verbose_any_leadtime(varname)

all_ordered_variable_names = np.array(all_ordered_variable_names)
########################################################################################
########################################################################################
########################################################################################


########################################################################################
### plot SIC observations and active grid cells ########################################
########################################################################################
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

sic, _, _ = dataloader.data_generation(pd.Timestamp("2014-01-01"))

fig, axs = plt.subplots(3, 4, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(sic[0, :, :, 11 - i], cmap="Blues", vmin=0, vmax=1)
    ax.imshow(land_edge, cmap="gray", alpha=0.5)
    ax.set_title(months[i])
    ax.axis("off")
fig.suptitle("SIC for 2013", fontsize=20)
#plt.savefig("icenet/experimental/figures/sic_2013.png", dpi=300)
plt.show()

active_grid_cells = []
for date in pd.date_range(start="2012-01-01", end="2012-12-01", freq="MS"):
    _, _, tmp = dataloader.data_generation(date)
    active_grid_cells.append(tmp[0, :, :, 0, 0])
active_grid_cells = np.array(active_grid_cells)

fig, axs = plt.subplots(3, 4, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(active_grid_cells[i, :, :], cmap="Blues")
    ax.imshow(land_edge, cmap="gray", alpha=0.5)
    ax.set_title(months[i])
    ax.axis("off")
fig.suptitle("Active grid cells for 2012", fontsize=20)
#plt.savefig("icenet/experimental/figures/active_grid_cells_2012.png", dpi=300)
plt.show()
########################################################################################
### plot spatial forecast ##############################################################
########################################################################################
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(forecast[:, :, i, :].sum(axis=-1), cmap="Blues")
    ax.imshow(land_edge, cmap="gray", alpha=0.5)
    ax.set_title(all_ordered_variable_names[i])
    ax.axis("off")
plt.show()
########################################################################################
########################################################################################
########################################################################################


########################################################################################
### plot receptive field etc. ##########################################################
########################################################################################

leadtime = 1

heatmap = img.sum(axis=(0))[:, :, leadtime-1]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(heatmap != 0, cmap="gray")
axs[0, 0].imshow(land_edge, cmap="gray", alpha=0.5)
axs[0, 0].set_title("Receptive field")
axs[0, 0].axis("off")

axs[0, 1].imshow(heatmap > 0, cmap="Reds")
axs[0, 1].imshow(land_edge, cmap="gray", alpha=0.5)
axs[0, 1].set_title("Values > 0")
axs[0, 1].axis("off")

axs[0, 2].imshow(heatmap < 0, cmap="Reds")
axs[0, 2].imshow(land_edge, cmap="gray", alpha=0.5)
axs[0, 2].set_title("Values < 0")
axs[0, 2].axis("off")

axs[1, 0].imshow((heatmap > 0)*(heatmap), cmap="Reds")
axs[1, 0].imshow(land_edge, cmap="gray", alpha=0.3)
axs[1, 0].set_title("Positive values")
axs[1, 0].axis("off")

axs[1, 1].imshow((heatmap < 0)*(-1*heatmap), cmap="Reds")
axs[1, 1].imshow(land_edge, cmap="gray", alpha=0.3)
axs[1, 1].set_title("Negative values")
axs[1, 1].axis("off")

axs[1, 2].imshow(np.abs(heatmap), cmap="Reds")
axs[1, 2].imshow(land_edge, cmap="gray", alpha=0.3)
axs[1, 2].set_title("Sum of absolute values")
axs[1, 2].axis("off")

plt.suptitle(f"September 2012 forecast, leadtime {leadtime} (all variables)")

#plt.savefig(f"./icenet/experimental/figures/receptive_field_leadtime_{leadtime}.png", dpi=300)

plt.show()
########################################################################################
########################################################################################
########################################################################################


########################################################################################
### plot sum of positive, negative and absolute feature importance for all leadtimes ###
########################################################################################
fig, (ax1, ax2, ax3) = plt.subplots(3, 6, figsize=(15, 10))

for i, ax in enumerate(ax1):    
    ax.imshow(np.abs(img.sum(axis=0))[:, :, 5-i], cmap="Reds", vmin=0, vmax=np.max(np.abs(img.sum(axis=0))))
    ax.imshow(land_edge, cmap="gray", alpha=0.3)
    ax.set_title(f"Leadtime {6-i}")
    ax.axis("off")

for i, ax in enumerate(ax2):    
    ax.imshow((img*(img > 0)).sum(axis=0)[:, :, 5-i], cmap="Reds", vmin=0, vmax=np.max(img.sum(axis=0)))
    ax.imshow(land_edge, cmap="gray", alpha=0.3)
    ax.set_title(f"Leadtime {6-i}")
    ax.axis("off")

for i, ax in enumerate(ax3):
    ax.imshow(np.abs((img*(img < 0))).sum(axis=0)[:, :, 5-i], cmap="Blues", vmin=0, vmax=np.max(img.sum(axis=0)))
    ax.imshow(land_edge, cmap="gray", alpha=0.3)
    ax.set_title(f"Leadtime {6-i}")
    ax.axis("off")

fig.suptitle("Average feature importance for September 2012 SIC prediction (absolute, positive, negative)", fontsize=16)
#plt.savefig("./icenet/experimental/figures/feature_importance_september_2012_global.png", dpi=300)

plt.show()
########################################################################################
########################################################################################
########################################################################################


########################################################################################
### plot spatial feature importance for all variables for a selected leadtime ##########
########################################################################################
leadtime = 1

fig, axes = plt.subplots(5, 10, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(np.abs(img[i, :, :, leadtime-1]), cmap="Reds", vmin=0, vmax=np.max(np.abs(img[:, :, :, leadtime-1])))
    ax.imshow(land_edge, cmap="gray", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    # set title with linebreaks
    ax.set_title(all_ordered_variable_names[i], fontsize=4)

fig.suptitle(f"Spatial feature importance for {leadtime} month leadtime for all variables", fontsize=16)

#plt.savefig(f"./icenet/experimental/figures/feature_importance_leadtime_{leadtime}_all_variables.png", dpi=600)

plt.show()
########################################################################################
########################################################################################
########################################################################################

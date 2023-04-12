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
from icenet.experimental.utils import load_icenet_model

land_mask = np.load(LAND_MASK_PATH)
land_edge = createLandEdgeImage()
model = load_icenet_model()


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

init_dates = pd.date_range(start="1980-01-01", end="2021-12-01", freq="YS")

for init_date in init_dates:
    inputs, _, _ = dataloader.data_generation(init_date)
    sics = inputs[0, :, :, :]

    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(sics[:, :, 11 - i], cmap="Blues", vmin=0, vmax=1)
        ax.imshow(land_edge, cmap="gray", alpha=0.5)
        ax.set_title(months[i])
        ax.axis("off")
    fig.suptitle(f"SIC for {init_date.year - 1}", fontsize=20)
    plt.savefig(f"icenet/experimental/figures/sic_{init_date.year - 1}.png", dpi=300)


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
### plot SIC forecasts for a specific month ############################################
########################################################################################


target_date = pd.DatetimeIndex(["2007-09-01"])
init_dates = pd.date_range(start=target_date - pd.DateOffset(months=6-1), end=target_date, freq="MS")

inputs_list = []
active_grid_cells_list = []
for forecast_start_date in init_dates:
    inputs, _, active_grid_cells = dataloader.data_generation(forecast_start_date)
    inputs_list.append(inputs[0])
    active_grid_cells_list.append(active_grid_cells)

ensemble_size = 10

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for leadtime in np.arange(1, 7):
    inputs = inputs_list[6 - leadtime]
    active_grid_cells = active_grid_cells_list[6 - leadtime]
    forecasts = []
    for _ in range(ensemble_size):
        forecasts.append(model(inputs[None, ...]))
    forecast = np.stack(forecasts, axis=0).mean(axis=0)[0, :, :, 2, leadtime - 1] * active_grid_cells[0, :, :, 0, leadtime - 1]
    ax = axs.flatten()[leadtime - 1]
    ax.imshow(forecast, cmap="Blues", vmin=0, vmax=1)
    ax.imshow(land_edge, cmap="gray", alpha=0.5)
    ax.set_title(f"Leadtime {leadtime}")
    ax.axis("off")
fig.suptitle("2007-09-01", fontsize=20)
#plt.savefig("icenet/experimental/figures/2007_09_01_predictions.png", dpi=300)
plt.show()

########################################################################################
### Create mask of variable regions ####################################################
########################################################################################

init_date_1 = pd.Timestamp("2007-07-01")
init_date_2 = pd.Timestamp("2013-07-01")
inputs_1, _, active_grid_cells_1 = dataloader.data_generation(init_date_1)
inputs_2, _, active_grid_cells_2 = dataloader.data_generation(init_date_2)

ensemble_size = 10
leadtime = 3

preds_1 = [model(inputs_1[0][None, ...]) for _ in range(ensemble_size)]
preds_1 = np.stack(preds_1, axis=0).mean(axis=0)[0, :, :, 2, leadtime] * active_grid_cells_1[0, :, :, 0, leadtime]
preds_2 = [model(inputs_2[0][None, ...]) for _ in range(ensemble_size)]
preds_2 = np.stack(preds_2, axis=0).mean(axis=0)[0, :, :, 2, leadtime] * active_grid_cells_2[0, :, :, 0, leadtime]

diff = preds_2 - preds_1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(preds_1, cmap="Blues")
ax1.imshow(land_edge, cmap="gray", alpha=0.5)
ax2.imshow(preds_2, cmap="Blues")
ax2.imshow(land_edge, cmap="gray", alpha=0.5)
ax3.imshow(diff > 0.6, cmap="Blues")
ax3.imshow(land_edge, cmap="gray", alpha=0.5)
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

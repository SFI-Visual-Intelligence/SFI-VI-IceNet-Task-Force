#%%
from interpret import (
    get_pred_and_explanations,
    # load_icenet_batch,
    # get_edge_mask,
    # unit_normalize,
)
import matplotlib.pyplot as plt
import os
import numpy as np

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

# import seaborn as sns
# import pandas as pd
# from einops import rearrange
from edge_detection import CannyEdgeDetector

LAND_MASK_PATH = "/Users/hjo109/Documents/GitHub/icenet-paper/icenet/analyze_input_influences/land_mask.npy"
REGION_MASK_PATH = "/Users/hjo109/Documents/GitHub/icenet-paper/icenet/analyze_input_influences/region_mask.npy"
VARIABLE_NAMES = [
    "sea_ice_concentration_01",
    "sea_ice_concentration_02",
    "sea_ice_concentration_03",
    "sea_ice_concentration_04",
    "sea_ice_concentration_05",
    "sea_ice_concentration_06",
    "sea_ice_concentration_07",
    "sea_ice_concentration_08",
    "sea_ice_concentration_09",
    "sea_ice_concentration_10",
    "sea_ice_concentration_11",
    "sea_ice_concentration_12",
    "linear_trend_sic_forecast_01",
    "linear_trend_sic_forecast_02",
    "linear_trend_sic_forecast_03",
    "linear_trend_sic_forecast_04",
    "linear_trend_sic_forecast_05",
    "linear_trend_sic_forecast_06",
    "2m_air_temp_anom_01",
    "2m_air_temp_anom_02",
    "2m_air_temp_anom_03",
    "500hp_air_temp_anom_01",
    "500hp_air_temp_anom_02",
    "500hp_air_temp_anom_03",
    "sea_surface_temp_anom_01",
    "sea_surface_temp_anom_02",
    "sea_surface_temp_anom_03",
    "downwelling_solar_rad_anom_01",
    "downwelling_solar_rad_anom_02",
    "downwelling_solar_rad_anom_03",
    "upwelling_solar_rad_anom_01",
    "upwelling_solar_rad_anom_02",
    "upwelling_solar_rad_anom_03",
    "sea_level_pressure_anom_01",
    "sea_level_pressure_anom_02",
    "sea_level_pressure_anom_03",
    "500hp_geopotential_height_anom_01",
    "500hp_geopotential_height_anom_02",
    "500hp_geopotential_height_anom_03",
    "150hp_geopotential_height_anom_01",
    "150hp_geopotential_height_anom_02",
    "150hp_geopotential_height_anom_03",
    "10hp_zonal_wind_speed_01",
    "10hp_zonal_wind_speed_02",
    "10hp_zonal_wind_speed_03",
    "x_wind_01",
    "y_wind_01",
    "land_mask",
    "cos_init_month",
    "sin_init_month",
]

#%%
# print(dict(zip(range(len(VARIABLE_NAMES)), VARIABLE_NAMES)))

# % initialize hydra (in interactive mode)
initialize(version_base=None, config_path="config", job_name="test")

#% Get config
cfg = compose(config_name="config")

# % Get edges of relevant regions to show in plots
# hudson_bay_mask = np.load(REGION_MASK_PATH) == 5

hudson_bay_system_mask = np.load(REGION_MASK_PATH) == 5
HUDSON_BAY_SYSTEM_EDGES = (
    CannyEdgeDetector(hudson_bay_system_mask[None, ...].astype(float)).detect()[0]
    / 255.0
)

#% Compute explanations
outputs, w, grads, grads_std, TIME_STAMP = get_pred_and_explanations(cfg)

LAND_MASK_EDGES = (
    CannyEdgeDetector(np.load(LAND_MASK_PATH)[None, ...].astype(float)).detect()[0]
    / 255.0
)

w_array = (w[..., 0] * 1).astype(float)
W_EDGES = CannyEdgeDetector(w_array).detect()[0] / 255.0

#% Setup paths
EXPERIMENT_PATH = f"figures/Exp{cfg.experiment_nr}_{cfg.predict_year}"
PREDICTIONS_PATH = os.path.join(EXPERIMENT_PATH, "prediction")
EXPLANATIONS_PATH = os.path.join(EXPERIMENT_PATH, "explanations")

os.makedirs(PREDICTIONS_PATH, exist_ok=True)
os.makedirs(EXPLANATIONS_PATH, exist_ok=True)

## save hydra configuration in experiment folder
OmegaConf.save(cfg, os.path.join(EXPERIMENT_PATH, "config.yaml"))

#%% Functions
def main():
    ########################################################################
    ## Perform experiments
    ########################################################################
    make_prediction_plots(outputs, TIME_STAMP, cfg.leadtime)
    feature_importance = plot_feature_importance(grads, grads_std, TIME_STAMP)
    # plot_feature_importance_location_maps(grads, grads_std, TIME_STAMP)


def make_prediction_plots(outputs, time_stamp, leadtime):
    plt.imshow(outputs[:, :, 0, leadtime])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 0, leadtime], dtype=float),
        alpha=LAND_MASK_EDGES,
        cmap="gray",
    )
    # plt.title("$SIC < 15 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(PREDICTIONS_PATH, f"sic_15_{time_stamp}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.imshow(outputs[:, :, 1, leadtime])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 1, leadtime], dtype=float),
        alpha=LAND_MASK_EDGES,
        cmap="gray",
    )
    # plt.title("$15 \% < SIC < 80 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(PREDICTIONS_PATH, f"sic_15_80_{time_stamp}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.imshow(outputs[:, :, 2, leadtime])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 2, leadtime], dtype=float),
        alpha=LAND_MASK_EDGES,
        cmap="gray",
    )
    # plt.title("$SIC > 80 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(PREDICTIONS_PATH, f"sic_80_{time_stamp}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.imshow(outputs[:, :, 1, leadtime] + outputs[:, :, 2, leadtime])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 1, leadtime], dtype=float),
        alpha=LAND_MASK_EDGES,
        cmap="gray",
    )
    # plt.title("$15 \% < SIC < 80 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(PREDICTIONS_PATH, f"sic_ice_{time_stamp}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def plot_feature_importance(grads, grads_std, time_stamp):
    """Plot mean importance over all features,"""
    # grad_pos = grads.copy()
    # grad_pos[grad_pos < 0] = 0

    # feature_importance = np.mean(grads, axis=(0, 1))
    feature_importance = np.max(np.abs(grads), axis=(0, 1))
    # sns.barplot(x=rearrange(feature_importance, ""), y=VARIABLE_NAMES)
    # sns.violinplot(feature_importance)

    num = np.sum((cfg.n_dropout_variations - 1) * grads_std**2)
    den = (
        cfg.n_dropout_variations * grads.shape[0] * grads.shape[1]
        - cfg.n_dropout_variations
    )

    feature_importance_std_pooled = np.sqrt(num / den)
    # feature_importance_std = np.mean(grads_std, axis=(0, 1))
    # table = pd.DataFrame(zip(VARIABLE_NAMES, feature_importance, feature_importance_std))
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.barh(
        range(len(VARIABLE_NAMES)),
        feature_importance,
        xerr=feature_importance_std_pooled,
        # align="center",
    )

    # plt.yticks(range(len(VARIABLE_NAMES)), [str(i) for i in VARIABLE_NAMES], rotation=-70)
    ax.set_yticks(
        range(len(VARIABLE_NAMES)), [str(i) for i in VARIABLE_NAMES]
    )  # , rotation=-70)
    ax.invert_yaxis()
    # plt.tick_params(axis="x", which="major", labelsize=5)
    plt.grid("on")
    # plt.tight_layout()
    plt.savefig(os.path.join(EXPLANATIONS_PATH, f"feature_importance_{time_stamp}.png"))
    plt.close()
    return feature_importance


def plot_feature_importance_location_maps(grads, grads_std, time_stamp):
    for i, name in enumerate(VARIABLE_NAMES):
        # plt.imshow(grads[..., i])
        # plt.colorbar()
        # plt.savefig(f"figures/Exp2/mean_pi_guided_backprop_{name}.png")
        # plt.close()

        grad_pos_max = np.max(grads[..., i])
        grad_pos_min = np.min(grads[..., i])

        blank = (np.ones_like(W_EDGES) + grad_pos_min) * grad_pos_max

        vmax = np.partition(grads[..., i].flatten(), -10)[-10]
        plt.imshow(grads[..., i], cmap="viridis", vmax=vmax)
        # plt.colorbar()
        # plt.imshow(blank, alpha=w_edges.astype(float), vmax=w_edges.max(), cmap="gray")

        ## Plot land mask edges
        plt.imshow(
            blank,
            alpha=LAND_MASK_EDGES.astype(float),
            vmax=LAND_MASK_EDGES.max() * 2,
            cmap="gray",
        )
        ## Plot hudson bay system edges
        plt.imshow(
            blank,
            alpha=HUDSON_BAY_SYSTEM_EDGES.astype(float),
            vmax=HUDSON_BAY_SYSTEM_EDGES.max() * 2,
            cmap="Blues",
        )

        # plt.imshow(blank, alpha=land_mask_edges.astype(float), vmax=land_mask_edges.max(), cmap="gray")
        # plt.imshow(blank, alpha=land_mask_edges.astype(float)/2, vmax=land_mask_edges.max()+1, cmap="gray")
        savepath = os.path.join(EXPLANATIONS_PATH, f"mean_{name}_{time_stamp}.png")
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.close()

        vmax = np.partition(grads_std[..., i].flatten(), -10)[-10]
        plt.imshow(grads_std[..., i], cmap="viridis")
        # plt.colorbar()
        # plt.imshow(
        #     blank, alpha=w_edges.astype(float), vmin=0, vmax=w_edges.max() + 1, cmap="gray"
        # )
        plt.imshow(
            blank,
            alpha=LAND_MASK_EDGES.astype(float),
            vmax=LAND_MASK_EDGES.max() * 2,
            cmap="gray",
        )
        # plt.imshow(blank, alpha=land_mask_edges.astype(float)/2, vmin=0, vmax=land_mask_edges.max()+1, cmap="gray")
        savepath = os.path.join(EXPLANATIONS_PATH, f"std_{name}_{time_stamp}.png")
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.close()

        grad_pos_standardized = grads[..., i] / (grads_std[..., i] + 1e-5)
        blank_standardized = (
            np.ones_like(W_EDGES) + grad_pos_standardized.min()
        ) * grad_pos_standardized.max()
        ## Lets send a mail and ask about the land mask.
        plt.imshow(grad_pos_standardized, cmap="viridis")
        # plt.colorbar()
        plt.imshow(
            blank_standardized, alpha=W_EDGES.astype(float), vmin=0, vmax=1, cmap="gray"
        )
        # plt.imshow(blank_standardized, alpha=land_mask_edges.astype(float)/2, vmin=0, vmax=land_mask_edges.max()+1, cmap="gray")
        savepath = os.path.join(EXPLANATIONS_PATH, f"mean_std{name}_{time_stamp}.png")
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.close()


#%% Run the code
if __name__ == "__main__":
    main()

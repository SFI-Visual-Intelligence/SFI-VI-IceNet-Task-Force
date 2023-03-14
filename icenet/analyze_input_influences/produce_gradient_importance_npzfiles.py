from interpret import get_pred_and_explanations
import matplotlib.pyplot as plt
import os
import numpy as np

# from hydra import initialize, compose
# from omegaconf import DictConfig, OmegaConf

# import seaborn as sns
# import pandas as pd
# from einops import rearrange
from utils.edge_detection import CannyEdgeDetector

# from munch import Munch
import wandb

LAND_MASK_PATH = (
    "/storage/experiments/icenet/icenet/analyze_input_influences/land_mask.npy"
)
REGION_MASK_PATH = (
    "/storage/experiments/icenet/icenet/analyze_input_influences/region_mask.npy"
)
DST_LOC_FOR_GRADS = "/storage/experiments/icenet/data/ordered_obs_npz/yearly_grads"
# LAND_MASK_PATH = "/Users/hjo109/Documents/GitHub/icenet-paper/icenet/analyze_input_influences/land_mask.npy"
# REGION_MASK_PATH = "/Users/hjo109/Documents/GitHub/icenet-paper/icenet/analyze_input_influences/region_mask.npy"
# DST_LOC_FOR_GRADS = "/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper/data/ordered_obs_npz/yearly_grads"

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


if not os.path.exists(DST_LOC_FOR_GRADS):
    os.makedirs(DST_LOC_FOR_GRADS)

# % Get edges of relevant regions to show in plots
# hudson_bay_mask = np.load(REGION_MASK_PATH) == 5

hudson_bay_system_mask = np.load(REGION_MASK_PATH) == 5
HUDSON_BAY_SYSTEM_EDGES = (
    CannyEdgeDetector(hudson_bay_system_mask[None, ...].astype(float)).detect()[0]
    / 255.0
)


class Cfg:
    """
    I would use munch for this, but the docker image is really not working with me here :( so this is a bad workaround.

    Do not do this at home! Use munch or DictConfig instead!
    """

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            setattr(self, k, v)

    def __str__(self):
        return f"Cfg({str(self.__dict__)})"

    def __repr__(self):
        return self.__str__()


#% Compute explanations

for year in range(1980, 2019):
    cfg = Cfg(
        dict(
            name="sum_of_thick_ice_in_hudson_bay",
            experiment_nr=22,
            predict_year=year,
            n_dropout_variations=25,
            mask_out_land_outputs=False,
            mask_out_land_in_loss=True,
            mask_out_land_grads=False,
            # leadtimes = [0, 1, 2, 3, 4, 5],
        )
    )

    for leadtime in range(6):
        print(f"Computing grads for year {cfg.predict_year} and leadtime {leadtime}")
        try:
            outputs, w, grads, grads_std, TIME_STAMP = get_pred_and_explanations(
                cfg, leadtime
            )
        except Exception as e:
            print(
                f"Failed to compute grads for year {cfg.predict_year} and leadtime {leadtime}"
            )
            # wandb.log({"error": str(e), "year": cfg.predict_year, "leadtime": leadtime})
            print(e)
            continue

        print("-" * 80)
        print(f"get_pred_and_explanations done for {TIME_STAMP}. Saving grads ...")
        print("-" * 80)
        ## Save grads with outputs, grads_std and timestamp
        np.savez(
            os.path.join(
                DST_LOC_FOR_GRADS, f"grads_{cfg.predict_year}_leadtime_{leadtime}.npz"
            ),
            outputs=outputs,
            grads=grads,
            grads_std=grads_std,
            TIME_STAMP=TIME_STAMP,
            # cfg=cfg,
        )
        print(
            f"Saved grads for year {cfg.predict_year} to {DST_LOC_FOR_GRADS}, for leadtime {leadtime}."
        )

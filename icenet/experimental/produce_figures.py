"""
This script is not in use.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))

from experimental.utils import get_pred_and_explanations
import matplotlib.pyplot as plt
import numpy as np
#from hydra import initialize, compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
#from edge_detection import CannyEdgeDetector
from experimental.make_hbs_map import createLandEdgeImage
import experimental.config as config
from collections import namedtuple

import importlib
importlib.reload(config)

print(dict(zip(range(len(config.VARIABLE_NAMES)), config.VARIABLE_NAMES)))

# initialize hydra (in interactive mode)
#initialize(version_base=None, config_path="icenet/experimental/conf", job_name="test")
# Get config
#cfg = compose(config_name="config", overrides=["experiment_nr=1", "batch_nr=0", "leadtime=0"])

CFG = namedtuple("Config", ["experiment_nr", "batch_nr", "leadtime", "mask_out_land_in_loss", "mask_out_land_outputs", "mask_out_land_grads", "n_dropout_variations"])
# Now initialize the config
cfg = CFG(1, 3, 0, True, True, True, 50)

# Compute explanations
#outputs, w, grads, grads_std, significant_mask, time_frame, cfg = get_pred_and_explanations(cfg)
outputs, w, grads, grads_std, significant_mask, cfg = get_pred_and_explanations(cfg)
time_stamp = time_frame[0].isoformat()

# Get edges of relevant regions to show in plots
# hudson_bay_mask = np.load(REGION_MASK_PATH) == 5
#land_mask_edges = (
#    CannyEdgeDetector(np.load(config.LAND_MASK_PATH)[None, ...].astype(float)).detect()[0] / 255.0
#)
w_array = (w[..., 0] * 1).astype(float)
#w_edges = CannyEdgeDetector(w_array).detect()[0] / 255.0
w_edges = createLandEdgeImage(w_array, method="sobel")


# Setup paths
experiment_path = f"figures/Exp{cfg.experiment_nr}"
predictions_path = os.path.join(experiment_path, "prediction")
explanations_path = os.path.join(experiment_path, "explanations")

os.makedirs(predictions_path, exist_ok=True)
os.makedirs(explanations_path, exist_ok=True)

## save hydra configuration in experiment folder
#OmegaConf.save(cfg, os.path.join(experiment_path, "config.yaml"))

# Functions to make plots
def make_prediction_plots(outputs, time_stamp):
    plt.imshow(outputs[:, :, 0, 0])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 0, 0], dtype=float), alpha=land_mask_edges, cmap="gray"
    )
    # plt.title("$SIC < 15 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(predictions_path, f"sic_15_{time_stamp}.png"), bbox_inches="tight", pad_inches=0
    )
    plt.close()

    plt.imshow(outputs[:, :, 1, 0])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 1, 0], dtype=float), alpha=land_mask_edges, cmap="gray"
    )
    # plt.title("$15 \% < SIC < 80 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(predictions_path, f"sic_15_80_{time_stamp}.png"), bbox_inches="tight", pad_inches=0
    )
    plt.close()

    plt.imshow(outputs[:, :, 2, 0])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 2, 0], dtype=float), alpha=land_mask_edges, cmap="gray"
    )
    # plt.title("$SIC > 80 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(predictions_path, f"sic_80_{time_stamp}.png"), bbox_inches="tight", pad_inches=0
    )
    plt.close()

    plt.imshow(outputs[:, :, 1, 0] + outputs[:, :, 2, 0])
    # plt.colorbar()
    plt.imshow(
        np.zeros_like(outputs[:, :, 1, 0], dtype=float), alpha=land_mask_edges, cmap="gray"
    )
    # plt.title("$15 \% < SIC < 80 \%$")
    plt.axis("off")
    plt.savefig(
        os.path.join(predictions_path, f"sic_ice_{time_stamp}.png"), bbox_inches="tight", pad_inches=0
    )
    plt.close()


def plot_feature_importance(grads, time_stamp):
    """Plot mean importance over all features, 
    """
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
    ax.set_yticks(range(len(VARIABLE_NAMES)), [str(i) for i in VARIABLE_NAMES]) #, rotation=-70)
    ax.invert_yaxis()
    # plt.tick_params(axis="x", which="major", labelsize=5)
    plt.grid("on")
    # plt.tight_layout()
    plt.savefig(os.path.join(explanations_path, f"feature_importance_{time_stamp}.png"))
    plt.close()
    return feature_importance


def plot_feature_importance_maps(grads, time_stamp):
    for i, name in enumerate(VARIABLE_NAMES):
        # plt.imshow(grads[..., i])
        # plt.colorbar()
        # plt.savefig(f"figures/Exp2/mean_pi_guided_backprop_{name}.png")
        # plt.close()

        grad_pos_max = np.max(grads[..., i])
        grad_pos_min = np.min(grads[..., i])

        blank = (np.ones_like(w_edges) + grad_pos_min) * grad_pos_max
        vmax = np.partition(grads[..., i].flatten(), -10)[-10]
        plt.imshow(grads[..., i], cmap="viridis", vmax=vmax)
        # plt.colorbar()
        # plt.imshow(blank, alpha=w_edges.astype(float), vmax=w_edges.max(), cmap="gray")
        plt.imshow(
            blank,
            alpha=land_mask_edges.astype(float),
            vmax=land_mask_edges.max() * 2,
            cmap="gray",
        )
        # plt.imshow(blank, alpha=land_mask_edges.astype(float), vmax=land_mask_edges.max(), cmap="gray")
        # plt.imshow(blank, alpha=land_mask_edges.astype(float)/2, vmax=land_mask_edges.max()+1, cmap="gray")
        savepath = os.path.join(explanations_path, f"mean_{name}_{time_stamp}.png")
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
            alpha=land_mask_edges.astype(float),
            vmax=land_mask_edges.max() * 2,
            cmap="gray",
        )
        # plt.imshow(blank, alpha=land_mask_edges.astype(float)/2, vmin=0, vmax=land_mask_edges.max()+1, cmap="gray")
        savepath = os.path.join(explanations_path, f"std_{name}_{time_stamp}.png")
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.close()

        grad_pos_standardized = grads[..., i] / (grads_std[..., i] + 1e-5)
        blank_standardized = (
            np.ones_like(w_edges) + grad_pos_standardized.min()
        ) * grad_pos_standardized.max()
        ## Lets send a mail and ask about the land mask.
        plt.imshow(grad_pos_standardized, cmap="viridis")
        # plt.colorbar()
        plt.imshow(
            blank_standardized, alpha=w_edges.astype(float), vmin=0, vmax=1, cmap="gray"
        )
        # plt.imshow(blank_standardized, alpha=land_mask_edges.astype(float)/2, vmin=0, vmax=land_mask_edges.max()+1, cmap="gray")
        savepath = os.path.join(explanations_path, f"mean_std{name}_{time_stamp}.png")
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
        plt.close()


# Run the code
if __name__ == "__main__":
    make_prediction_plots(outputs, time_stamp)
    feature_importance = plot_feature_importance(grads, time_stamp)
    plot_feature_importance_maps(grads, time_stamp)

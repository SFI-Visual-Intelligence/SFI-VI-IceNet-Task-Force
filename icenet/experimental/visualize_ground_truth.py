import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))  # if using jupyter kernel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_ID", type=str, default="2021_06_15_1854_icenet_nature_communications")
parser.add_argument("--forecast_date", type=str, default="2013-09-01")
parser.add_argument("--land_mask_path", type=str, default="./icenet/experimental/masks/land_mask.npy")
parser.add_argument("--output_path", type=str, default="./icenet/experimental/figures")
args = parser.parse_args()

if __name__ == "__main__":
    dataloader_config_fpath = os.path.join(config.dataloader_config_folder, args.dataloader_ID + ".json")
    dataloader = utils.IceNetDataLoader(dataloader_config_fpath)

    forecast_start_date = pd.DatetimeIndex([args.forecast_date])[0] + pd.DateOffset(months=1)
    inputs, _, _ = dataloader.data_generation(forecast_start_date)

    land_mask = np.load(args.land_mask_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    plt.imshow((1 - land_mask), cmap="gray", alpha=1.0)
    plt.imshow(inputs[0, :, :, 0], cmap="Blues", alpha=0.9)
    plt.axis("off")
    plt.savefig(os.path.join(args.output_path, f"{args.forecast_date}_input.png"), dpi=300, bbox_inches="tight")

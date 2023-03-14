#%%
"""
Description: This script is used to get the ordered (unshuffled) data samples of the 
    data along with their dates. This is done by subclassing the IceNetDataloader class 
    and overriding all the shuffling behavior. The idea is to return the data along with dates
    such that it is easier to intrpret the results of the model.

Author: Harald Lykke Joakimsen
Mail: harald.l.joakimsen@uit.no
"""
import os

# os.chdir("/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper")

import config
from utils import IceNetDataLoader
import numpy as np
import json


dataloader_ID = "2023_ordered_printout_config"
dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)


class OrderedIceNetDataLoader(IceNetDataLoader):
    def __init__(self, dataloader_config_fpath, seed: int = 42):
        """
        Params:
        dataloader_config_fpath (str): Path to the data loader configuration
            settings JSON file, defining IceNet's input-output data configuration.

        seed (int): Random seed used for shuffling the training samples before
            each epoch.
        """

        with open(dataloader_config_fpath, "r") as readfile:
            self.config = json.load(readfile)

        if seed is None:
            self.set_seed(self.config["default_seed"])
        else:
            self.set_seed(seed)

        self.do_transfer_learning = False

        self.set_obs_forecast_IDs(dataset="train")
        self.set_transfer_forecast_IDs()
        self.all_forecast_IDs = self.obs_forecast_IDs
        self.remove_missing_dates()
        self.set_variable_path_formats()
        self.set_number_of_input_channels_for_each_input_variable()
        self.load_polarholes()
        self.determine_tot_num_channels()
        self.on_epoch_end()

        if self.config["verbose_level"] >= 1:
            print("Setup complete.\n")

    def __getitem__(self, batch_idx):
        """
        Generate one batch of data of size `batch_size` at batch index `batch_idx`
        into the set of batches in the epoch.
        """

        batch_start = batch_idx * self.config["batch_size"]
        batch_end = np.min(
            [(batch_idx + 1) * self.config["batch_size"], len(self.all_forecast_IDs)]
        )

        sample_idxs = np.arange(batch_start, batch_end)
        batch_IDs = [self.all_forecast_IDs[sample_idx] for sample_idx in sample_idxs]

        return self.data_generation(batch_IDs), batch_IDs

    def on_epoch_end(self):
        pass


if __name__ == "__main__":
    dataloader = OrderedIceNetDataLoader(dataloader_config_fpath)
    for (x, y, w), timestamp in dataloader:
        # x, y, w = dataloader[i]
        # if timestamp[0].month != 3:
        #     continue
        np.savez_compressed(
            os.path.join(
                config.ordered_obs_data_folder, f"yearly_samples/data_{timestamp[0].isoformat()}_to_{timestamp[-1].isoformat()}.npz"
            ),
            x=x,
            y=y,
            w=w,
            timestamp=timestamp[0].isoformat(),
        )
        print(f"Saved val data for time: {timestamp[0].isoformat()}.")


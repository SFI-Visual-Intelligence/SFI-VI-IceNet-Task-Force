"""
Description: This script is used to get the unshuffled or ordered data samples of the data along with dates.
    This is done by subclassing the IceNetDataloader class and overriding all the
    shuffling behavior. The idea is to return the data along with dates
    such that it is easier to intrpret the results of the model.

Author: Harald Lykke Joakimsen
Mail: harald.l.joakimsen@uit.no
"""
import os
import icenet.config as config
from icenet.utils import IceNetDataLoader
import numpy as np


dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)


class OrderedIceNetDataLoader(IceNetDataLoader):
    def __init__(self, dataloader_config_fpath):
        super().__init__(dataloader_config_fpath)

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


dataloader = OrderedIceNetDataLoader(dataloader_config_fpath)

# Get dates matched with indices
idx_table = np.zeros(len(dataloader.all_forecast_IDs), dtype=[("idx", int), ("date", str)])
for i, (_, batch_IDs) in enumerate(dataloader):
    for j, batch_ID in enumerate(batch_IDs):
        idx_table[batch_ID] = (i, batch_IDs)

np.savez_compressed(
    os.path.join(config.networks_folder, f"idx_table.npz"),
    idx_table=idx_table,
)


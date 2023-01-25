import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))

import pandas as pd
import numpy as np
import utils
import config
from tqdm import tqdm

destionation_folder = "icenet/experimental/results"


def create_dummy_dataframe():
    """
    Create a dummy dataframe with the same structure as the permute-and-predict results dataframe.
    """
    all_ordered_variable_names = dataloader.determine_variable_names()

    n_forecast_months = 6
    heldout_start = "2012-01-01"
    heldout_end = "2012-12-01"

    leadtimes = np.arange(1, n_forecast_months + 1)
    model_numbers = range(1)
    target_dates = pd.date_range(start=heldout_start, end=heldout_end, freq="MS")

    multi_index = pd.MultiIndex.from_product(
        [all_ordered_variable_names, model_numbers, leadtimes, target_dates],
        names=["Model number", "Leadtime", "Forecast date", "Variable"],
    )

    results_df = pd.DataFrame(
        index=multi_index, columns=["Feature importance"], dtype=np.float32
    )

    for model_number in tqdm(model_numbers):
        for _, varname in enumerate(tqdm(all_ordered_variable_names, leave=False)):
            for target_date in target_dates:
                for leadtime in leadtimes:
                    results_df.loc[
                        model_number, leadtime, target_date, varname
                    ] = np.random.uniform(0, 100)

    return results_df


if __name__ == "__main__":

    dataloader_ID = "2021_06_15_1854_icenet_nature_communications"
    dataloader_config_fpath = os.path.join(
        config.dataloader_config_folder, dataloader_ID + ".json"
    )
    dataloader = utils.IceNetDataLoader(dataloader_config_fpath)

    os.makedirs(destionation_folder, exist_ok=True)

    data_frame = create_dummy_dataframe()
    data_frame.to_csv(os.path.join(destionation_folder, "dummy_dataframe.csv"))

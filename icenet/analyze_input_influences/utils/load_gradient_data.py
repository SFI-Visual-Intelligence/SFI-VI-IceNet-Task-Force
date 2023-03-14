import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import logging
import pandas as pd

DATA_PATH = "/Users/hjo109/Library/CloudStorage/OneDrive-UiTOffice365/need_to_clean/Documents/GitHub/icenet-paper/data/ordered_obs_npz/yearly_grads"
log = logging.getLogger(__name__)


def get_grads_from_year(year, leadtime):
    """
    Load grads and grads_std from a given year and leadtime.
    """
    ## first get the data from the first month of the year (january) from the current year
    ## then get the rest from the next year (since weird time lag in the data)
    out_data = dict(
        TIME_STAMP=[None, None],
        grads=np.empty((12, 432, 432, 50)),
        grads_std=np.empty((12, 432, 432, 50)),
        outputs=np.empty((12, 432, 432, 3, 6)),
    )

    ## Get data for january and february
    filename = f"grads_{year}_leadtime_{leadtime}.npz"
    filepath = os.path.join(DATA_PATH, filename)
    try:
        data_jan = np.load(filepath)
    except Exception as e:
        log.warning(
            "failed to load data for year "
            + str(year)
            + " and leadtime "
            + str(leadtime)
        )

    if year == 1980:
        log.info("actual time stamp for start" + str(data_jan["TIME_STAMP"]) + "+2")
    else:
        log.info("actual time stamp for start" + str(data_jan["TIME_STAMP"]) + "+11")

    ## Fill in the data
    out_data["TIME_STAMP"] = [
        datetime.datetime(year, 1, 1),
        datetime.datetime(year, 12, 31),
    ]
    if (
        year == 1980
    ):  ## special case for 1980 since we do not have data from before 1980-02-01
        out_data["grads"][0, ...] = np.nan
        out_data["grads_std"][0, ...] = np.nan
        out_data["outputs"][0, ...] = np.nan
        out_data["TIME_STAMP"][0] = datetime.datetime(year, 2, 1)
    else:
        out_data["grads"][0, ...] = data_jan["grads"][11:, ...]
        out_data["grads_std"][0, ...] = data_jan["grads_std"][11:, ...]
        out_data["outputs"][0, ...] = data_jan["outputs"][11:, ...]

    ## Get data for the rest of the year
    filename = f"grads_{year+1}_leadtime_{leadtime}.npz"
    filepath = os.path.join(DATA_PATH, filename)
    try:
        data_rest = np.load(filepath)
    except Exception as e:
        log.warning(
            "failed to load data for year "
            + str(year)
            + " and leadtime "
            + str(leadtime)
        )
        log.info("actual time stamp for end" + str(data_rest["TIME_STAMP"]) + "+10")

    ## Fill in the data
    out_data["grads"][1:, ...] = data_rest["grads"][:11, ...]
    out_data["grads_std"][1:, ...] = data_rest["grads_std"][:11, ...]
    out_data["outputs"][1:, ...] = data_rest["outputs"][:11, ...]
    return out_data


def test_get_data_from_year():
    years = list(range(1980, 2019))
    leadtimes = list(range(1, 5))
    year_leadtimes = np.zeros((len(years), len(leadtimes)))

    for i, year in enumerate(years):
        for j, leadtime in enumerate(leadtimes):
            try:
                get_data_from_year(year, leadtime)
                year_leadtimes[i, j] = 1
            except Exception as e:
                print(f"failed for year {year} and leadtime {leadtime}")
                log.exception(e)

    completion_message = (
        "all data loaded"
        if np.sum(year_leadtimes) == len(years) * len(leadtimes)
        else "some data missing"
    )
    print(completion_message)

    # plt.imshow(year_leadtimes)
    # plt.savefig("year_leadtime_missing.png")


if __name__ == "__main__":
    test_get_data_from_year()

#!/bin/bash

python -V >> log.txt
echo ----------------- >> log.txt
echo getting started >> log.txt
python icenet/gen_masks.py >> log.txt
echo icenet/gen_masks.py finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
python icenet/download_sic_data.py >> log.txt
echo icenet/download_sic_data.py finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
./download_era5_data_in_parallel.sh >> log.txt
echo download_era5_data_in_parallel.sh finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
./download_cmip6_data_in_parallel.sh >> log.txt
echo download_cmip6_data_in_parallel.sh finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
./rotate_wind_data_in_parallel.sh >> log.txt
echo rotate_wind_data_in_parallel.sh finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
./download_seas5_forecasts_in_parallel.sh >> log.txt
echo download_seas5_forecasts_in_parallel.sh finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
python icenet/biascorrect_seas5_forcasts.py >> log.txt
echo icenet/biascorrect_seas5_forcasts.py finished >> log.txt
echo ----------------- >> log.txt

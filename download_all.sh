#!/bin/bash

python -V >> log.txt
echo ----------------- >> log.txt
echo getting started >> log.txt
python icenet/gen_masks.py >> log.txt
echo icenet/gen_masks.py finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
python icenet/download_sic_data.py
echo icenet/download_sic_data.py finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
./download_era5_data_in_parallel.sh
echo download_era5_data_in_parallel.sh finished >> log.txt
echo ----------------- >> log.txt

######### Probably not needed (did not contribute with a lot in performance and is a lot of data.) #############
# echo ----------------- >> log.txt
# ./download_cmip6_data_in_parallel.sh
# echo download_cmip6_data_in_parallel.sh finished >> log.txt
# echo ----------------- >> log.txt

################################################################################################################
echo ----------------- >> log.txt
./rotate_wind_data_in_parallel.sh
echo rotate_wind_data_in_parallel.sh finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
./download_seas5_forecasts_in_parallel.sh
echo download_seas5_forecasts_in_parallel.sh finished >> log.txt
echo ----------------- >> log.txt

echo ----------------- >> log.txt
python icenet/biascorrect_seas5_forecasts.py >> log.txt
echo icenet/biascorrect_seas5_forecasts.py finished >> log.txt
echo ----------------- >> log.txt

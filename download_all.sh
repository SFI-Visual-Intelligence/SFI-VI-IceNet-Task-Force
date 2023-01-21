#!/bin/bash

touch logs/download_all.log

echo ----------------- >> logs/download_log.txt
echo ----------------- >> logs/download_log.txt

echo getting started >> logs/download_log.txt
python icenet/gen_masks.py >> logs/download_log.txt
echo icenet/gen_masks.py finished >> logs/download_log.txt
echo ----------------- >> logs/download_log.txt

python icenet/download_sic_data.py >> logs/download_log.txt
echo icenet/download_sic_data.py finished >> logs/download_log.txt
echo ----------------- >> logs/download_log.txt

./download_era5_data_in_parallel.sh
echo download_era5_data_in_parallel.sh finished >> logs/download_log.txt
echo ----------------- >> logs/download_log.txt

######### Ignore for now. Small contribution in performance, but much data. #############
# ./download_cmip6_data_in_parallel.sh
# echo download_cmip6_data_in_parallel.sh finished >> logs/download_log.txt
# echo ----------------- >> logs/download_log.txt

#########################################################################################

./rotate_wind_data_in_parallel.sh
echo rotate_wind_data_in_parallel.sh finished >> logs/download_log.txt
echo ----------------- >> logs/download_log.txt

./download_seas5_forecasts_in_parallel.sh
echo download_seas5_forecasts_in_parallel.sh finished >> logs/download_log.txt
echo ----------------- >> logs/download_log.txt

python icenet/biascorrect_seas5_forecasts.py >> logs/download_log.txt
echo icenet/biascorrect_seas5_forecasts.py finished >> logs/download_log.txt
echo ----------------- >> logs/download_log.txt

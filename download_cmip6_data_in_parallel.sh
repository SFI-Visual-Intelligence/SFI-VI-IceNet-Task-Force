#!/bin/bash

## Not sure if this is needed here to
python copy_cdsapirc_file_to_container_home.py

mkdir -p logs/cmip6_download_logs/ &&

# # MRI-ESM2.0
# ############ THIS HOLTS FOR SOME REASON ##############
# python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r2i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/MRI_r2i1p1f1.txt 
######################################################

#######################################################################################################################################
######################################################### New try #####################################################################

# # python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r1i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/MRI_r1i1p1f1.txt
# # python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r2i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/MRI_r2i1p1f1.txt
# python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r3i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/MRI_r3i1p1f1.txt
# # python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r4i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/MRI_r4i1p1f1.txt
# # python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r5i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/MRI_r5i1p1f1.txt

# # # EC-Earth3
# # python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r2i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/EC_r2i1p1f1.txt
# # python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r7i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/EC_r7i1p1f1.txt
# # python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r10i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/EC_r10i1p1f1.txt
# # python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r12i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/EC_r12i1p1f1.txt
# # python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r14i1p1f1 2>&1 | tee -a logs/cmip6_download_logs/EC_r14i1p1f1.txt 

#######################################################################################################################################
#######################################################################################################################################

# #######################################################################################################################################
# ######################################################### Old #########################################################################

# MRI-ESM2.0
python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r1i1p1f1 > logs/cmip6_download_logs/MRI_r1i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r2i1p1f1 > logs/cmip6_download_logs/MRI_r2i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r3i1p1f1 > logs/cmip6_download_logs/MRI_r3i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r4i1p1f1 > logs/cmip6_download_logs/MRI_r4i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id MRI-ESM2-0 --member_id r5i1p1f1 > logs/cmip6_download_logs/MRI_r5i1p1f1.txt 2>&1 &

# EC-Earth3
python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r2i1p1f1 > logs/cmip6_download_logs/EC_r2i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r7i1p1f1 > logs/cmip6_download_logs/EC_r7i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r10i1p1f1 > logs/cmip6_download_logs/EC_r10i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r12i1p1f1 > logs/cmip6_download_logs/EC_r12i1p1f1.txt 2>&1 &
python icenet/download_cmip6_data.py  --source_id EC-Earth3 --member_id r14i1p1f1 > logs/cmip6_download_logs/EC_r14i1p1f1.txt 2>&1 &

# #######################################################################################################################################
# #######################################################################################################################################
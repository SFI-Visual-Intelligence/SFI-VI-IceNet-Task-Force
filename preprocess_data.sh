echo processing data starting >> log.txt && \
python3 icenet/gen_data_loader_config.py && \
echo "icenet/gen_data_loader_config.py done" >> log.txt && \
python3 icenet/preproc_icenet_data.py && \
echo "icenet/preproc_icenet_data.py done" >> log.txt && \
## Making the program faster
# python3 icenet/gen_numpy_obs_train_val_datasets.py && \
# echo "icenet/gen_numpy_obs_train_val_datasets.py done" >> log.txt && \
# python3 icenet/gen_tfrecords_obs_train_val_datasets.py && \
# echo "icenet/gen_tfrecords_obs_train_val_datasets.py done" >> log.txt && \
echo "Preprocessing done!" >> log.txt

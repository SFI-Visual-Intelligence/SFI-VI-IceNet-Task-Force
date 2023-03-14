# Interpretability direction (`analyze_input_influences`)
Idea is to print out data in batches of 1, pick out all dates where we are predicting SIC
of September (the hardest month) from March, apply NoiceGrad or similar and find out if there is a consistent pattern to it. Are there places for which we should pay more attention and take measurements more often when predicting far into the future?
  1. Created `2023_ordered_printout_config.json` in `dataloader_configs` defining batch size of 1.
  2. Made `get_ordered_data_samples.py` in `icenet` folder that defines the task of iterating through 
    data and storing each march sample along with the active cells (area deemed relevant for predictions) 
    and the timeframe for the data.
  3. Defined file `interpret.py` which defines functions to perform guided backprop for dropout monte carlo versions of given model for all data. 
  4. (--Copy back to laptop--) Way too slow. Change of plan: used `produce_gradient_importance_npzfiles.py` (which uses `interpret.py`) instead run on the cluster, then copied npz-files in `yearly_grads` back to laptop.
  5. Finally, aggregate the results and make visualizations with `make_figures_from_yearly_grads.ipynb` locally.


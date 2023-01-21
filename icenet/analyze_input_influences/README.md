# Interpretability direction (`analyze_input_influences`)
- Idea is to print out data in batches of 1, pick out all dates where we are predicting SIC
of September (the hardest month) from March, apply NoiceGrad or similar and find out if there is a consistent pattern to it. Are there places for which we should pay more attention and take measurements more often when predicting far into the future?
  1. Created `2023_ordered_printout_config.json` in `dataloader_configs` defining batch size of 1.
  2. Made `get_ordered_data_samples.py` in `icenet` folder that defines the task of iterating through 
    data and storing each march sample along with the active cells (area deemed relevant for predictions) 
    and the timeframe for the data.
  3. Copy back to laptop
  4. Defined file `interpret.py` which defines functions to perform guided backprop for dropout monte carlo versions of given model for all data. 
  5. `produce_figures.py` uses interpret and `edge_detection.py` to makes estimates and show figures.


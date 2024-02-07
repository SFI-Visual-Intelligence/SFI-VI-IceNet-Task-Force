import tensorflow as tf

# Assert that we have a GPU available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
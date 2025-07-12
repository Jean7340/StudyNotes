import torch

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

# If using GPU, print the name of the GPU
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


import tensorflow as tf

# List available physical devices
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Show all devices TensorFlow can see
print("All devices:", tf.config.list_physical_devices())

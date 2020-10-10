import pandas as pd
import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
    
import torch
# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
# Reading dataframe 
directory = '.../documents_in_txt/'

data_sus_rank = pd.read_csv(directory+'dataframe.txt', sep=',', index_col=0, encoding='utf-8')

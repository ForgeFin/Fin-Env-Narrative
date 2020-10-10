import pandas as pd
import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

## Import Bert and pytorch
from transformers import BertTokenizer, BertModel, BertConfig

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
    
    
############################################
''' Specify directory and load dataframe '''
############################################

directory = '.../dataframe_folder/'
documents = '.../documents_as_txt/'

data_sus_rank = pd.read_csv(directory+'dataframe.txt', sep=',', index_col=0, encoding='utf-8')


#  Load pre-trained model tokenizer (vocabulary)
print("bert-base-uncased 12-layer, 768-hidden, 12-heads, 110M parameters.")
print("Trained on lower-cased English text.") 

pretrained_weights = 'bert-base-uncased'

# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertModel.from_pretrained(pretrained_weights)
#model.eval() #model in evaluation mode by default.. does not need to be active
model.cuda() # Tell pytorch to run this model on the GPU.

# Tokenize input
to_tokenize = data_sus_rank['fname']
to_tokenize = to_tokenize.to_list()


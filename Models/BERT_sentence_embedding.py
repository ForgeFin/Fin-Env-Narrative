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
documents = '.../documents_as_txt/'  # specify directory and adjust x = re.findall(r'documents_as_txt/(.*)',file) in ''' conversion to numpy ''' to according folder name
save_drive = '.../Bert_docs_after_tokenizing/' # specify where encoded txt-files should be saved

data_sus_rank = pd.read_csv(directory+'dataframe.txt', sep=',', index_col=0, encoding='utf-8')

data_sus_rank['path'] = documents
data_sus_rank['fileload'] = data_sus_rank[['path', 'fname']].apply(lambda x: ''.join(x), axis=1)

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
to_tokenize = data_sus_rank['fileload']
to_tokenize = to_tokenize.to_list()


##########################################################################
''' add special tokens [CLS] and [SEP] and convert each token to an ID '''
##########################################################################

start_time = time.time() 
tester = []
file_numb = []
file_no = 1
length_files = []
length_of_seq = 1000  # sentence number that will be used of file
print("Sentence length used:", length_of_seq)
max_numword_per_sentence = 200 ## due to limit of Berts sequence processing (a sequence cannot be longer than 512)

for file in to_tokenize:
    longst = []
    x = re.findall(r'documents_as_txt/(.*)',file)  ## << HERE: adjust 'documents_as_txt' to your folder-name containing the txt-files 
    name = x[0]
    with open(file, encoding='utf-8') as f:
        length_file = len(f.readlines()) # whatsoever closes file; length of sentence
        
    with open(file, encoding='utf-8') as f:
        with open(save_drive+name, 'a+', encoding='utf-8') as wrfile:
            if length_file < length_of_seq:
                longst.extend(f.readline() for i in range(length_file)) # uses length of file
                for line in longst:
                    tokenized_text = tokenizer.tokenize(line)
                    if len(tokenized_text) > max_numword_per_sentence:
                       print("Sentence too long")
                       y = line.split()
                       firstpart = y[0:max_numword_per_sentence]
                       firstpart = ' '.join(firstpart)
                       test = tokenizer.encode(firstpart, add_special_tokens=True)
                       test_str = ','.join([str(elem) for elem in test])
                       wrfile.write(test_str + '\n')
                    else:
                       test = tokenizer.encode(line, add_special_tokens=True)
                       test_str = ','.join([str(elem) for elem in test])
                       wrfile.write(test_str + '\n')

            else:
                longst.extend(f.readline() for i in range(length_of_seq)) 
                for line in longst:
                    tokenized_text = tokenizer.tokenize(line)
                    if len(tokenized_text) > max_numword_per_sentence:
                       print("Too long")
                       y = line.split()
                       firstpart = y[0:max_numword_per_sentence]
                       firstpart = ' '.join(firstpart)
                       test = tokenizer.encode(firstpart, add_special_tokens=True)
                       test_str = ','.join([str(elem) for elem in test])
                       wrfile.write(test_str + '\n')
                    else:
                       test = tokenizer.encode(line, add_special_tokens=True)
                       test_str = ','.join([str(elem) for elem in test])
                       wrfile.write(test_str + '\n')
    file_no += 1    
    print(file_no)
    
print("")
print("")
print("Sequence/Document lenght is trimmed to a maximum of %s lines per file" % length_of_seq)
print("If sentence is longer than %s words, sentence is trimmed down to 150 by .split()" % max_numword_per_sentence)
print("Tokenizer.encode then might split 150 words into more than 150")
print("")
print("Each file is now encoded -- each token is converted to an ID and for sentences a special token is added")


end_time = time.time() - start_time 
print("--- %s minutes ---" % (((time.time() - start_time))/60))

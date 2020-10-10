#######################################################################################################################
''' 
This code requires a GPU. 
For example, you could use Google Colab and integrate a GPU (see https://colab.research.google.com/notebooks/gpu.ipynb )

Code works as follows:
1. Load texts with one sentence per line (lower case words)
2. For each text, use first 200 tokens of each sentence and encode tokens to IDs
3. Save ID-encoded files to a folder
4. Reload each file and do a feed-forward pass to BERT
5. Retrieve vector representation of CLS-tokens for each sentence from BERT and save each document to np.array
'''
#######################################################################################################################


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
save_drive = '.../Bert_docs_after_tokenizing/' # specify where ID encoded txt-files should be saved
path = '.../numpy_arrays/' # specify where numpy arrays should be saved

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

max_numword_per_sentence = 200  ## due to limit of Berts sequence processing (a sequence cannot be longer than 512)
                                ## Note that even if you limit the number of words to 200, the BERT tokenizer could
                                ## split a word into several tokens, e.g. "playing", will be split into "play" and "##ing"


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


######################################################################################################
''' do a feed-forward pass of each encoded file and save CLS-vector representation as numpy array  '''
######################################################################################################

print("load file list of encoded files")
bert_test_files=[]
for ff in os.listdir(save_drive):
    if ff.endswith(".txt"):
        bert_test_files.append(os.path.join(save_drive,ff))

## Model
## Careful, needs about 880 min for 8772 files!!!
cutter = 100 # cutter creates chunks, cause otherwise to hard to process a 1000 sentence-file 

start_time = time.time()

cnt = 1
for i in bert_test_files:
    longst = []

    x = re.findall(r'Bert_Docs_ID_1000sents_maxWord200_longstsent510/(.*)',i)  # für google colab umdrehen 'Bert_Docs_tokenized_encoded/'
    x = re.findall(r'.*[^.txt]',x[0])
    name = x[0]

    with open(i, encoding='utf-8') as f:
        length_file = len(f.readlines()) # whatsoever closes file; length of sentence
        print("     length of file %s" % length_file)
    with open(i, encoding='utf-8') as est:
        if length_file < cutter:
            ll = [] ## used for appending sentences of each document
            for line in est:
                seperated = [int(i) for i in line.split(',')] # line is "string,string" etc. splitted at "," and converted to number
                seperated = seperated + [0] * (longest_sentence - len(seperated))
                ll.append(seperated)
            bert_line = pd.DataFrame(ll)
            padded = np.array(bert_line)
            attention_mask = np.where(padded != 0, 1, 0)

            # each padded document is converted into a tensor
            input_ids = torch.tensor(padded)
            input_ids_gpu = input_ids.to(device) 
            attention_mask = torch.tensor(attention_mask)
            attention_mask_gpu = attention_mask.to(device)
                
            ### Runing each document separatly through BERT works ONLY if BERT is not fine-tuned!!!
            with torch.no_grad():
                last_hidden_states = model(input_ids_gpu.long(), attention_mask=attention_mask_gpu.long())
            features = last_hidden_states[0][:,0,:].cpu().numpy()
        
        else:
            longst.extend(est.readline() for i in range(length_file))
            chunks = [longst[x:x+100] for x in range(0, len(longst), 100)]
            start = 0
            print("      Doc splitted into %s subchunks" % len(chunks))
            for element in chunks:
                ll = [] ## needs to be here, since otherwise every chunk is appended seperately
                for line in element:
                    seperated = [int(i) for i in line.split(',')] # line is "string,string" etc. splitted at "," and converted to number
                    seperated = seperated + [0] * (longest_sentence - len(seperated))
                    ll.append(seperated)
                bert_line = pd.DataFrame(ll)
                padded = np.array(bert_line)
                attention_mask = np.where(padded != 0, 1, 0)

                # each padded document is converted into a tensor
                input_ids = torch.tensor(padded)
                input_ids_gpu = input_ids.to(device) 
                attention_mask = torch.tensor(attention_mask)
                attention_mask_gpu = attention_mask.to(device)
                
                if start == 0:
                    ### Runing each chunk separatly through BERT works ONLY if BERT is not fine-tuned!!!
                    with torch.no_grad():
                        last_hidden_states = model(input_ids_gpu.long(), attention_mask=attention_mask_gpu.long())
                    features = last_hidden_states[0][:,0,:].cpu().numpy()
                    start +=1

                else:
                    with torch.no_grad():
                        last_hidden_states = model(input_ids_gpu.long(), attention_mask=attention_mask_gpu.long())
                    features_append = last_hidden_states[0][:,0,:].cpu().numpy()
                    features = np.vstack((features,features_append))

    np.save(path+name+'.npy',features) ## path needs to be defined, saves each features to npy format
    cnt += 1

    print("file no %s from total %s" % (cnt, len(bert_test_files)))
    #print("from total %s " % len(bert_test_files))

end_time = time.time() - start_time 
print("--- %s minutes ---" % (((time.time() - start_time))/60))


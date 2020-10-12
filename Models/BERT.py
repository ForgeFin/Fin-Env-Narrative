###########################################
''' (Code-)Author: Felix Armbrust, 2020 ''' 
###########################################

import os
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt

from string import punctuation
from nltk.corpus import stopwords
from collections import Counter

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score 
from multiprocessing import cpu_count
        
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, GRU, Bidirectional, Concatenate, TimeDistributed, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from keras.utils import to_categorical, Sequence
from keras.utils import plot_model
from keras.engine.topology import Layer, InputSpec
from keras import initializers

import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop

from keras.metrics import TruePositives, TrueNegatives, FalseNegatives, FalsePositives
from keras import regularizers
from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#############################################################################
''' Define custom metrics for calculating F-Score, precision, and recall;
    Create generator for loading files '''
#############################################################################

def calc_metrics(yprediction, ytrue):
    ''' calculating precision '''
    TruePositive = 0
    for i in range(len(yprediction)):
        if yprediction[i] == ytrue[i] and yprediction[i] == 1:
            TruePositive += 1
    
    FalsePositive = 0 # outcome where the model incorrectly predicts the positive class
    for i in range(len(yprediction)):
        if yprediction[i] != ytrue[i] and yprediction[i] == 1:
            FalsePositive += 1
    
    FalseNegative = 0 # outcome where the model incorrectly predicts the negative class
    for i in range(len(yprediction)):
        if ytrue[i] != yprediction[i] and yprediction[i] == 0:
            FalseNegative += 1
            
    precision = TruePositive / (TruePositive + FalsePositive + K.epsilon())
    
    recall = TruePositive / (FalseNegative + TruePositive + K.epsilon())
    
    Fscore = 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    return Fscore, precision , recall#,TruePositive, FalsePositive, FalseNegative#,


def TP_FP_FN_forBootstrap(yprediction, ytrue):
    ''' calculating precision '''
    TruePositive_list = []
    for i in range(len(yprediction)):
        if yprediction[i] == ytrue[i] and yprediction[i] == 1:
            TruePositive = 1
        else:
            TruePositive =0
        TruePositive_list.append(TruePositive)
    
    FalsePositive_list = []
    for i in range(len(yprediction)):
        if yprediction[i] != ytrue[i] and yprediction[i] == 1:
            FalsePositive = 1
        else:
            FalsePositive = 0
        FalsePositive_list.append(FalsePositive)
    
    FalseNegative_list = []
    for i in range(len(yprediction)):
        if ytrue[i] != yprediction[i] and yprediction[i] == 0:
            FalseNegative = 1
        else:
            FalseNegative = 0
        FalseNegative_list.append(FalseNegative)
            
    diction = {'TP':TruePositive_list,'FP':FalsePositive_list,'FN':FalseNegative_list}
    df = pd.DataFrame(diction, columns=['TP','FP','FN'])
    
    return df
  

class F1History(tf.keras.callbacks.Callback):
    ###  See Marco Cerliani: https://stackoverflow.com/questions/61683829/calculating-fscore-for-each-epoch-using-keras-not-batch-wise
    def __init__(self, train, validation=None):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):

        logs['F1_score_train'] = float('-inf')
        X_train, y_train = self.train[0], self.train[1]
        y_pred = (self.model.predict(X_train).ravel()>0.5)+0
        score = f1_score(y_train, y_pred)       

        if (self.validation):
            logs['F1_score_val'] = float('-inf')
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_val_pred = (self.model.predict(X_valid).ravel()>0.5)+0
            val_score = f1_score(y_valid, y_val_pred)
            logs['F1_score_train'] = np.round(score, 5)
            logs['F1_score_val'] = np.round(val_score, 5)
        else:
            logs['F1_score_train'] = np.round(score, 5)
            
     
print("Creating Attention Layer")
# Humbold AttLayer
# See: https://humboldt-wi.github.io/blog/research/information_systems_1819/group5_han/

class AttentionLayer(Layer):
    # Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification (2016) Yang et. al.
    # Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    # Theano backend
    
    def __init__(self,attention_dim=100,return_coefficients=False,**kwargs):
        # Initializer 
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform') # initializes values with uniform distribution
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Builds all weights
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)),name='W')
        self.b = K.variable(self.init((self.attention_dim, )),name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)),name='u')
        self.trainable_weights = [self.W, self.b, self.u]

        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, hit, mask=None):
        # Here, the actual calculation is done
        uit = K.bias_add(K.dot(hit, self.W),self.b)
        uit = K.tanh(uit)
        
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        
        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = hit * ait
        
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]


class final_generator(Sequence):
    '''
    Generates data for Keras
    list_IDs =  a list of npy. files to load
    labels   =  a dictionary of labels {'filename1.npy':1,'filename1.npy':0,...etc}
    filepath =  for example '.../testing_generator/'
    '''
    
    def __init__(self, list_IDs, labels, filepath, batch_size=32, sentence_length=1000, features=768 ,shuffle=True, to_fit=True):
        ''' initialization '''
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.sentence_length = sentence_length 
        self.features = features 
        self.filepath = filepath
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()
        
    def __len__(self):
        ''' Denotes the number of batches per epoch '''
        
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        ''' 
        Generate one batch of data
        :param index: index of the batch; is created when called!
        :return: X and y when fitting. X only when predicting
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._generate_data(list_IDs_temp)

        if self.to_fit:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # suffles list IN PLACE! so does NOT create new list 
            
            
    def _generate_data(self, list_IDs_temp):
        '''
        Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        
        list_IDs_temp is created when __getitem__ is called
        
        '''
        # Initialization
        X = np.empty((self.batch_size, self.sentence_length, self.features))
        y = np.empty((self.batch_size), dtype=int)
        
        for i, ID in enumerate(list_IDs_temp):
            # i is a number;
            # ID is the file-name
               
            # load single file
            single_file = np.load(os.path.join(self.filepath,ID))
            ## create empty array to contain batch of features and labels
            batch_features = np.zeros((self.sentence_length, self.features))
            
            #####
            # to allow for shorter than 1000-sentences
            single_file = single_file[:self.sentence_length,:self.features]
            
            #####
            
            # pad loaded array to same length        
            shape = np.shape(single_file)
            batch_features[:shape[0],:shape[1]] = single_file 
            
            ## append to sequence
            X[i,] = batch_features
            
            y[i] = self.labels[ID] ### this looks-up according rating (Note ID = file name.npy)
            
        return X, y


#############################################################################
''' Define your directory containing all documents'''
#############################################################################

bertdocs = '.../Bert_Docs/' ## directory containing the npy-files (!)
save_path = '.../save_results/' ## directory with datafile containin the label data and file IDs (i.e., filenames)

saver = '.../train_test_val_split/' ## directory (if needed) to save unique train, val, test split

## specify where to save final models 
## Note: Create subfolders with labelnames
pic_path = '.../Paper_final/' ## directory to save model results
savemodel = pic_path ## if needed specify


###########################################
''' >>>   Split unique val, test, train '''
###########################################

from sklearn.model_selection import train_test_split

data_all_rank = pd.read_csv(save_path+'/dataframe.txt', sep=',', index_col=0, encoding='utf-8')


validation_all = pd.read_csv(saver+'validation_data_rs0.txt', sep=',', index_col=0, encoding='utf-8')
train_all = pd.read_csv(saver+'train_data_rs0.txt', sep=',', index_col=0, encoding='utf-8')
test_all = pd.read_csv(saver+'test_data_rs0.txt', sep=',', index_col=0, encoding='utf-8')

validation_all['path'] = bertdocs
train_all['path'] = bertdocs
test_all['path'] = bertdocs

validation_all['np_name'] = validation_all['fname'].str.replace('txt','npy', n=1) # need to replace npy ending with txt for merging
train_all['np_name'] = train_all['fname'].str.replace('txt','npy', n=1) # need to replace npy ending with txt for merging
test_all['np_name'] = test_all['fname'].str.replace('txt','npy', n=1) # need to replace npy ending with txt for merging

validation_all['fileload'] = validation_all[['path', 'np_name']].apply(lambda x: ''.join(x), axis=1)
train_all['fileload'] = train_all[['path', 'np_name']].apply(lambda x: ''.join(x), axis=1)
test_all['fileload'] = test_all[['path', 'np_name']].apply(lambda x: ''.join(x), axis=1)


''' Adding market cap and industry '''
validation_all = validation_all.merge(data_all_rank[['Log_MarketCap','osha_gov_SIC_numeric','osha_gov_SIC_manual','fname']], on=['fname'], how='inner')
train_all = train_all.merge(data_all_rank[['Log_MarketCap','osha_gov_SIC_numeric','osha_gov_SIC_manual','fname']], on=['fname'], how='inner')
test_all = test_all.merge(data_all_rank[['Log_MarketCap','osha_gov_SIC_numeric','osha_gov_SIC_manual','fname']], on=['fname'], how='inner')

print("Note: to_categorical starts with 0, so need to substract 1 from industry codification")
industry_train = keras.utils.to_categorical(np.array(train_all["osha_gov_SIC_numeric"]-1))
print("See shape industry_train",industry_train.shape)
industry_val = keras.utils.to_categorical(np.array(validation_all["osha_gov_SIC_numeric"]-1))
industry_test = keras.utils.to_categorical(np.array(test_all["osha_gov_SIC_numeric"]-1))

mktcap_test = np.array(test_all["Log_MarketCap"])
mktcap_train = np.array(train_all["Log_MarketCap"])
mktcap_val = np.array(validation_all["Log_MarketCap"])

# save these to list and check unique counts
x_test, y_test_EP, y_test_bhar, y_test_ep90, y_test_eps = test_all['fileload'].to_list(), test_all['env_t'].to_list(), test_all['BHAR_t'].to_list(), test_all['env_q'].to_list(), test_all['EPS_q'].to_list()
x_train, y_train_EP, y_train_bhar, y_train_ep90, y_train_eps = train_all['fileload'].to_list(), train_all['env_t'].to_list(), train_all['BHAR_t'].to_list(), train_all['env_q'].to_list(), train_all['EPS_q'].to_list()
x_val, y_val_EP, y_val_bhar, y_val_ep90, y_val_eps = validation_all['fileload'].to_list(), validation_all['env_t'].to_list(), validation_all['BHAR_t'].to_list(), validation_all['env_q'].to_list(), validation_all['EPS_q'].to_list()


############################################################
''' Create first model for financial label (direct path) '''
############################################################

label = 'BHAR_t'       ## << HERE 'BHAR_t', 'EPS_q'

y_train = y_train_bhar ## << HERE  y_train_bhar, y_train_eps
y_test = y_test_bhar   ## << HERE   y_test_bhar, y_test_eps
y_val = y_val_bhar     ## << HERE   y_val_bhar, y_val_eps


label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)

epoch = 10
batch = 32
drop_cnn = 0.5
regularization = 0.01  ## check if l1, l1_l2 or just l2 regularization!
dense_size = 50
drop_end = 0.5

no_sentences_per_doc = 1000 #1000
sentence_embedding = 768  # if just normal seq
# Create a zip object from two lists

print("For", label)
print("Test size %s, Train size %s, Val size %s" % (len(y_test),len(y_train),len(y_val)))
y_train, y_test, y_val = np.array(y_train, dtype='float32'), np.array(y_test, dtype='float32'), np.array(y_val, dtype='float32')

X_train, X_test, X_val = np.array(x_train, dtype='str'), np.array(x_test, dtype='str'), np.array(x_val, dtype='str')

y_label_train = dict(zip(X_train,y_train)) # labels is actual input for generator
y_label_val = dict(zip(X_val,y_val))       # labels is actual input for generator
y_label_test = dict(zip(X_test,y_test))    # labels is actual input for generator


def create_model(loadweight=None):
    ##### MODEL #####
    sequence_input  = Input(shape=(no_sentences_per_doc, sentence_embedding))

    gru_layer = Bidirectional(GRU(50, #activation='tanh',
                              return_sequences=True#True
                              ))(sequence_input)
    
    ### consider putting here a TimeDistributed Dense layer... l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    ### TimeDistributed applies a layer to every temporal slice of an input; input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension
    sent_dense = Dense(100, activation='relu', name='sent_dense')(gru_layer)  # make signal stronger
    
    sent_att,sent_coeffs = AttentionLayer(100,return_coefficients=True,name='sent_attention')(sent_dense)
    sent_att = Dropout(0.5,name='sent_dropout')(sent_att)

    preds = Dense(1, activation='sigmoid',name='output')(sent_att)  # NOTE: SIGMOID FOR 2-CLASS PROBLEM; softmax for multiclass problem
    model = Model(sequence_input, preds)
            


    if loadweight is not None:
        model.load_weights(loadweight)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[TruePositives(name='true_positives'),
                          TrueNegatives(name='true_negatives'),
                          FalseNegatives(name='false_negatives'),
                          FalsePositives(name='false_positives'),
                          ])
    return model

##########################
''' Calculate Baseline '''
##########################

unique, counts = np.unique(y_train,                 
                           return_counts=True)
baseline_trains = dict(zip(unique, counts))
total = baseline_trains[1] + baseline_trains[0]
total_precision_train = baseline_trains[1] / total
f_baseline_train = 2*(total_precision_train)/(1+total_precision_train)
print("Baseline_train:",baseline_trains)

unique_val, counts_val = np.unique(y_val,
                                   return_counts=True)
baseline_vals = dict(zip(unique_val, counts_val))
total_val = baseline_vals[1] + baseline_vals[0]
total_precision_val = baseline_vals[1] / total_val
f_baseline_val = 2*(total_precision_val)/(1+total_precision_val)
print("Baseline_val:",baseline_vals)

unique_test, counts_test = np.unique(y_test,
                                     return_counts=True)
baseline_test = dict(zip(unique_test, counts_test))
total_test = baseline_test[1] + baseline_test[0]
total_precision_test = baseline_test[1] / total_test
f_baseline_test = 2*(total_precision_test)/(1+total_precision_test)
print("Baseline_test:",baseline_test)



#################
''' Run model '''
#################

accuracy, val_accuracy, manual_acc = [], [], []
loss, val_loss = [], []
fscores, f_m, f_m_val = [], [], []

precision_sc, precision_sc_val = [], []
recall_sc, recall_sc_val = [], []

f1_sc, f1_sc_val = [], []

tp, fp, fn = [], [], []
tp_val, fp_val, fn_val = [], [], []

# other variable that must be fed is a dictionary of labels
training_generator = final_generator(X_train, y_label_train, bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                     features=sentence_embedding, shuffle=False, to_fit=True)
validation_generator = final_generator(X_val, y_label_val, bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                       features=sentence_embedding, shuffle=False, to_fit=True)  # to_fit returns X and y


#print('Testing %s label..' % labelname)
model = create_model()

# checkpoint
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), verbose=1, save_best_only=False, save_weights_only=True, mode='auto')#, save_freq='epoch')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=training_generator, 
                              #steps_per_epoch=None,   # if unspecified, will use the len(generator) as a number of steps.
                              epochs=epoch,
                              validation_data=validation_generator,
                              #use_multiprocessing=True,
                              #workers=num_workers, 
                              callbacks=callbacks_list)   

loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])

loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()


#####################
''' Reload models '''
#####################

model_loads = []
for root, dirs, files in os.walk(savemodelpath):
    for file in files:
        if file.endswith(".hdf5"):  
            model_loads.append(os.path.join(root,file))


Fscore_val, Fscore_train = [], []

fscorepredict_val, fscorepredict_train = [], []
fscorepredict_val_custom, fscorepredict_val_sklearn = [], []

recall_train, recall_val = [], []
precision_train, precision_val = [], []

validation_generator_predict = final_generator(X_val, y_label_val, bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                               features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y
    
training_generator_predict = final_generator(X_train, y_label_train, bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                             features=sentence_embedding, shuffle=False, to_fit=False)

###  Note: 
#    Only samples that are equally divided by batch_size are used, i.e.,
#    If batch=32 and 7104 training samples, 7104 / 32 = 222 all samples are used
#    But for unequal sets like test = 878, 878/32 = 27,43 it neglects 878-(32*27)= 14 samples
#    Hence, for Fscore calculaton y_test must be trimmed!
#   
#    However, it does not make a difference if using calc_metrics, as calc_metrics uses only whats passed


cnt = 0
for i in model_loads:  
    percent = cnt / epoch
    print("Progress: ",percent)
    ## recreating model
    model = create_model(loadweight=i)

    # for validation data
    y_pred = model.predict_generator(validation_generator_predict, verbose=0)
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    
    ## see Careful-note
    yy = len(val_preds)
    cm, p, r = calc_metrics(val_preds, y_val[0:yy])#y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    
    cm = f1_score(y_val[0:yy], val_preds)
    #cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)
    
    # for training data
    y_pred =  model.predict_generator(training_generator_predict, verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)
    cnt += 1


# saving loss and fscore for plots
diction = {'loss':loss,'val_loss':val_loss,'fscore':fscorepredict_train,'val_fscore':fscorepredict_val_custom}
df = pd.DataFrame(diction, columns=['loss','val_loss','fscore','val_fscore'])
df.to_csv(savemodelpath+label+'loss_fscore.txt', sep=',', encoding='utf-8')

diction = {'pr_train':precision_train,'pr_val':precision_val,'rec_train':recall_train,'rec_val':recall_val}
df2 = pd.DataFrame(diction, columns=['pr_train','pr_val','rec_train','rec_val'])
df2.to_csv(savemodelpath+label+'recall_precision.txt', sep=',', encoding='utf-8')

fscorepredict_train = np.array(fscorepredict_train)
fscorepredict_val_custom = np.array(fscorepredict_val_custom)

best_validation_model = np.argmax(fscorepredict_val_custom)

best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)

f1_sc = np.insert(fscorepredict_train, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(fscorepredict_val_custom, 0, f_baseline_val, axis=0)


test_generator_predict = final_generator(X_test, y_label_test, bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                             features=sentence_embedding, shuffle=False, to_fit=False)

# run best model (highest validation score)
print("Reloading best validation model")
loadpath_best = model_loads[best_validation_model]
model = create_model(loadweight=loadpath_best)

y_pred = model.predict_generator(test_generator_predict)

test_preds = [1 if x > 0.5 else 0 for x in y_pred]
yy = len(test_preds)

f_test,pr_test, re_test = calc_metrics(test_preds, y_test[0:yy])

bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test[0:yy])
bootstrap_test.to_csv(savemodelpath+label+'bootstrap_test.txt',sep=',',encoding='utf-8')


# save best train_preds and val_preds
loadpath_best = model_loads[best_train_index]
print("Reloading best training model")
model = create_model(loadweight=loadpath_best)

y_pred_train = model.predict_generator(training_generator_predict)
train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
f_train,pr_train, re_train = calc_metrics(train_preds, y_train)

saved_train_preds = pd.DataFrame(train_preds)
saved_train_preds.to_csv(savemodelpath+label+'saved_train_preds.txt',sep=',',encoding='utf-8')


loadpath_best = model_loads[best_val_index]
model = create_model(loadweight=loadpath_best)

y_pred_val = model.predict_generator(validation_generator_predict)
val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
yy = len(val_preds)
f_val,pr_val, re_val = calc_metrics(val_preds, y_val[0:yy])

saved_val_preds = pd.DataFrame(val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_val_preds.txt',sep=',',encoding='utf-8')

bootstrap_val = TP_FP_FN_forBootstrap(val_preds,y_val[0:yy])
bootstrap_val.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_val.txt',sep=',',encoding='utf-8')


########################
''' Get model output '''
########################

#### PLOT USES F1_SC since this is the correct Fscore 
plt.plot(range(0,epoch+1), f1_sc, color='blue', marker='o', markersize=5)
plt.plot(range(0,epoch+1), f1_sc_val, color='green', marker='s', markersize=5, linestyle ='--')
plt.plot(0, f1_sc[0], color='red', marker='o', markersize=5)
plt.plot(0, f1_sc[0], color='red', marker='x', markersize=10)
plt.annotate('Always guessing class 1', xy=(0, f1_sc[0]),xytext=(0.2,f1_sc[0]))
plt.plot(best_train_index+1, f1_sc[best_train_index+1], color='orange', marker='o', markersize=6)
plt.plot(best_val_index+1, f1_sc_val[best_val_index+1], color='orange', marker='o', markersize=6)
plt.annotate('Always guessing class 1', xy=(0, f1_sc_val[0]), xytext=(0.2,f1_sc_val[0]))

plt.plot(0, f1_sc[0], color='red', marker='o', markersize=5)
plt.plot(0, f1_sc_val[0], color='red', marker='o', markersize=5)
plt.title('model f1')
plt.ylabel('f1')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower right')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_f1.png')
plt.close()

plt.plot(range(1,epoch+1), loss, color='blue', marker='o', markersize=5)
plt.plot(range(1,epoch+1), val_loss, color='green', marker='s', markersize=5, linestyle ='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='upper right')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_loss.png')
plt.close()


plt.plot(range(1,epoch+1), precision_train, color='blue', marker='s', markersize=5)
plt.plot(range(1,epoch+1), precision_val, color='green', marker='s', markersize=5, linestyle ='--')
plt.plot(best_train_index+1, precision_train[best_train_index], color='orange', marker='o', markersize=5)
plt.plot(best_val_index+1, precision_val[best_val_index], color='orange', marker='o', markersize=5)
plt.title('Model')
plt.ylabel('Precision')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train_precision','validation_precision'], loc='upper left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_precision.png')
plt.close()


plt.plot(range(1,epoch+1), recall_train, color='blue', marker='o', markersize=5, linestyle =':')
plt.plot(range(1,epoch+1), recall_val, color='green', marker='s', markersize=5, linestyle ='-')
plt.plot(best_train_index+1, recall_train[best_train_index], color='orange', marker='o', markersize=5)
plt.plot(best_val_index+1, recall_val[best_val_index], color='orange', marker='o', markersize=5)
plt.title('Model')
plt.ylabel('Recall')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train_recall','val_recall'], loc='upper left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_recall.png')
plt.close()


print("")
print("Baseline train:", baseline_trains)
print("Baseline test:", baseline_test)
print("Baseline val:", baseline_vals)
print("")
print("Epochs:",epoch)
print("Label:",label)
print("Batch %s, Dropout of CNNs %s, Regularization %s, Dense Size %s, Drpout at Combined %s" % (batch,drop_cnn,regularization,dense_size,drop_end))

print("")
print("                         Precision, Recall,  Fscore,   tstat")
print("overall_Baseline(train): %f, %f, %f" % (total_precision_train, 1,f_baseline_train))
print("overall_Baseline(val):   %f, %f, %f" % (total_precision_val, 1,f_baseline_val))
print("overall_Baseline(test):  %f, %f, %f" % (total_precision_test, 1,f_baseline_test))
print("")
print("")
print("(train):                 %f, %f, %f" % (precision_train[best_train_index], recall_train[best_train_index],f1_sc[best_train_index+1]))
print("(val):                   %f, %f, %f" % (precision_val[best_val_index],recall_val[best_val_index],f1_sc_val[best_val_index+1]))
print("")
print("(test):                  %f, %f, %f" % (pr_test,re_test,f_test))
print("")
print("")
print("")
print(">>      Copy in txt format!")



###################################################################
''' Define new generator to control for Market Cap and Industry '''
###################################################################

class multi_input_generator(Sequence):
    ''' 
    Generates data for Keras
    
    list_IDs =  a list of npy. files to load
    labels   =  a dictionary of labels {'filename1.npy':1,'filename1.npy':0,...etc}
    filepath =  for example '.../testing_generator/'
    '''
    
    def __init__(self, list_IDs, mkt_cap, indstry, labels, filepath, batch_size=32, sentence_length=1000, features=768, ind_dim=8, shuffle=True, to_fit=True):
        ''' initialization '''
        self.list_IDs = list_IDs
        self.mkt_cap = mkt_cap
        self.indstry = indstry
        self.labels = labels
        self.batch_size = batch_size
        self.sentence_length = sentence_length 
        self.features = features
        self.ind_dim = ind_dim
        self.filepath = filepath
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()
        
    def __len__(self):
        ''' Denotes the number of batches per epoch '''
        
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        ''' 
        Generate one batch of data
        :param index: index of the batch; is created when called!
        :return: X and y when fitting. X only when predicting
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._generate_data(list_IDs_temp)
        

        if self.to_fit:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # suffles list IN PLACE! so does NOT create new list 
            
            
    def _generate_data(self, list_IDs_temp):
        '''
        Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        
        list_IDs_temp is created when __getitem__ is called
        
        '''
        # Initialization
        X = np.empty((self.batch_size, self.sentence_length, self.features))
        y = np.empty((self.batch_size), dtype=int)
        
        market = np.empty((self.batch_size), dtype=int)
        industry = np.empty((self.batch_size, self.ind_dim), dtype=int) ## note: industry dim set to 8
        
        for i, ID in enumerate(list_IDs_temp):
            # i is a number;
            # ID is the file-name
               
            # load single file
            single_file = np.load(os.path.join(self.filepath,ID))
            ## create empty array to contain batch of features and labels
            batch_features = np.zeros((self.sentence_length, self.features))
            
            #####
            # to allow for shorter than 1000-sentences
            single_file = single_file[:self.sentence_length,:self.features]
            
            #####
            
            # pad loaded array to same length        
            shape = np.shape(single_file)
            batch_features[:shape[0],:shape[1]] = single_file 
            
            ## append to sequence
            X[i,] = batch_features
            
            industry[i,] = self.indstry[ID]
            market[i] = self.mkt_cap[ID]
            
            y[i] = self.labels[ID] ### this looks-up according rating (Note ID = file name.npy)
            
        return [X, market, industry], y


########################################################
''' Create model for environmental labels (path env) '''
########################################################

label = 'env_t'    ## << HERE 'env_t','env_q'

y_train = y_train_EP ## << HERE  y_train_EP, y_train_ep90
y_test = y_test_EP   ## << HERE  y_test_EP, y_test_ep90
y_val = y_val_EP     ## << HERE  y_val_EP, y_val_ep90

label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)

epoch = 10
batch = 32
drop_cnn = 0.5
regularization = 0.01  ## check if l1, l1_l2 or just l2 regularization!
dense_size = 50
drop_end = 0.5

no_sentences_per_doc = 1000 #1000
sentence_embedding = 768  # if just normal seq

# Create a zip object from two lists
print("For", label)
print("Test size %s, Train size %s, Val size %s" % (len(y_test),len(y_train),len(y_val)))
y_train, y_test, y_val = np.array(y_train, dtype='float32'), np.array(y_test, dtype='float32'), np.array(y_val, dtype='float32')

X_train, X_test, X_val = np.array(x_train, dtype='str'), np.array(x_test, dtype='str'), np.array(x_val, dtype='str')

y_label_train = dict(zip(X_train,y_train)) # labels is actual input for generator
y_label_val = dict(zip(X_val,y_val))       # labels is actual input for generator
y_label_test = dict(zip(X_test,y_test))    # labels is actual input for generator


mktcap_label_train = dict(zip(X_train,mktcap_train)) # labels is actual input for generator
mktcap_label_val = dict(zip(X_val,mktcap_val))       # labels is actual input for generator
mktcap_label_test = dict(zip(X_test,mktcap_test))    # labels is actual input for generator

ind_label_train = dict(zip(X_train,industry_train)) # labels is actual input for generator
ind_label_val = dict(zip(X_val,industry_val))       # labels is actual input for generator
ind_label_test = dict(zip(X_test,industry_test))    # labels is actual input for generator


industry_dim = industry_train.shape[1]

def create_model(loadweight=None):
    ##### MODEL #####
    sequence_input  = Input(shape=(no_sentences_per_doc, sentence_embedding))

    ### ADDING MARKET CAP AND INDUSTRY
    market_cap = Input(shape=(1,), dtype='float32')
    industry = Input(shape=(industry_dim,), dtype='float32')
    
    gru_layer = Bidirectional(GRU(50, #activation='tanh',
                              return_sequences=True#True
                              ))(sequence_input)
    
    sent_dense = Dense(100, activation='relu', name='sent_dense')(gru_layer)  # make signal stronger
    
    sent_att,sent_coeffs = AttentionLayer(100,return_coefficients=True,name='sent_attention')(sent_dense)
    sent_att = Dropout(0.5,name='sent_dropout')(sent_att)

    ## combine market cap and industry with cnn
    combined = Concatenate()([sent_att, market_cap, industry])

    preds = Dense(1, activation='sigmoid',name='output')(combined)  # NOTE: SIGMOID FOR 2-CLASS PROBLEM; softmax for multiclass problem
    model = Model([sequence_input,market_cap,industry], preds)

    if loadweight is not None:
        model.load_weights(loadweight)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[TruePositives(name='true_positives'),
                          TrueNegatives(name='true_negatives'),
                          FalseNegatives(name='false_negatives'),
                          FalsePositives(name='false_positives'),
                          ])
    return model


#######################
''' Create baseline '''
#######################

unique, counts = np.unique(y_train,                 
                           return_counts=True)
baseline_trains = dict(zip(unique, counts))
total = baseline_trains[1] + baseline_trains[0]
total_precision_train = baseline_trains[1] / total
f_baseline_train = 2*(total_precision_train)/(1+total_precision_train)
print("Baseline_train:",baseline_trains)

unique_val, counts_val = np.unique(y_val,
                                   return_counts=True)
baseline_vals = dict(zip(unique_val, counts_val))
total_val = baseline_vals[1] + baseline_vals[0]
total_precision_val = baseline_vals[1] / total_val
f_baseline_val = 2*(total_precision_val)/(1+total_precision_val)
print("Baseline_val:",baseline_vals)

unique_test, counts_test = np.unique(y_test,
                                     return_counts=True)
baseline_test = dict(zip(unique_test, counts_test))
total_test = baseline_test[1] + baseline_test[0]
total_precision_test = baseline_test[1] / total_test
f_baseline_test = 2*(total_precision_test)/(1+total_precision_test)
print("Baseline_test:",baseline_test)


###################
''' Model setup '''
###################

accuracy, val_accuracy, manual_acc = [], [], []
loss, val_loss = [], []
fscores, f_m, f_m_val = [], [], []

precision_sc, precision_sc_val = [], []
recall_sc, recall_sc_val = [], []

f1_sc, f1_sc_val = [], []

tp, fp, fn = [], [], []
tp_val, fp_val, fn_val = [], [], []


# other variable that must be fed is a dictionary of labels
training_generator = multi_input_generator(X_train, mktcap_label_train, ind_label_train, y_label_train,
                                           bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                           features=sentence_embedding, shuffle=False, to_fit=True)
validation_generator = multi_input_generator(X_val, mktcap_label_val, ind_label_val, y_label_val,
                                             bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                             features=sentence_embedding, shuffle=False, to_fit=True)  # to_fit returns X and y

#print('Testing %s label..' % labelname)
model = create_model()


# checkpoint
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), verbose=1, save_best_only=False, save_weights_only=True, mode='auto')#, save_freq='epoch')
callbacks_list = [checkpoint]


history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator, ## << HERE
                              epochs=epoch,
                              callbacks=callbacks_list
                              )


loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])

loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()


#####################
''' Reload models '''
#####################

model_loads = []
for root, dirs, files in os.walk(savemodelpath):
    for file in files:
        if file.endswith(".hdf5"):  
            model_loads.append(os.path.join(root,file))


Fscore_val, Fscore_train = [], []

fscorepredict_val, fscorepredict_train = [], []
fscorepredict_val_custom, fscorepredict_val_sklearn = [], []

recall_train, recall_val = [], []
precision_train, precision_val = [], []


# other variable that must be fed is a dictionary of labels
training_generator_predict = multi_input_generator(X_train, mktcap_label_train, ind_label_train, y_label_train,
                                           bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                           features=sentence_embedding, shuffle=False, to_fit=False)
validation_generator_predict = multi_input_generator(X_val, mktcap_label_val, ind_label_val, y_label_val,
                                             bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                             features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y


cnt = 0
for i in model_loads:  
    percent = cnt / epoch
    print("Progress: ",percent)
    ## recreating model
    model = create_model(loadweight=i)

    # for validation data
        
    y_pred = model.predict_generator(validation_generator_predict, verbose=0)
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    
    ## see Careful-note
    yy = len(val_preds)
    cm, p, r = calc_metrics(val_preds, y_val[0:yy])#y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    
    cm = f1_score(y_val[0:yy], val_preds)
    #cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)

    # for training data
    y_pred =  model.predict_generator(training_generator_predict, verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)
    cnt += 1

# saving loss and fscore for plots
diction = {'loss':loss,'val_loss':val_loss,'fscore':fscorepredict_train,'val_fscore':fscorepredict_val_custom}
df = pd.DataFrame(diction, columns=['loss','val_loss','fscore','val_fscore'])
df.to_csv(savemodelpath+label+'loss_fscore.txt', sep=',', encoding='utf-8')

diction = {'pr_train':precision_train,'pr_val':precision_val,'rec_train':recall_train,'rec_val':recall_val}
df2 = pd.DataFrame(diction, columns=['pr_train','pr_val','rec_train','rec_val'])
df2.to_csv(savemodelpath+label+'recall_precision.txt', sep=',', encoding='utf-8')


fscorepredict_train = np.array(fscorepredict_train)
fscorepredict_val_custom = np.array(fscorepredict_val_custom)


best_validation_model = np.argmax(fscorepredict_val_custom)

best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)

f1_sc = np.insert(fscorepredict_train, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(fscorepredict_val_custom, 0, f_baseline_val, axis=0)


test_generator_predict = multi_input_generator(X_test, mktcap_label_test, ind_label_test, y_label_test,
                                             bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                             features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y

# run best model (highest validation score)
print("Reloading best validation model")
loadpath_best = model_loads[best_validation_model]
model = create_model(loadweight=loadpath_best)

y_pred = model.predict_generator(test_generator_predict)

test_preds = [1 if x > 0.5 else 0 for x in y_pred]
yy = len(test_preds)

f_test,pr_test, re_test = calc_metrics(test_preds, y_test[0:yy])


bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test[0:yy])
bootstrap_test.to_csv(savemodelpath+label+'bootstrap_test.txt',sep=',',encoding='utf-8')


# save best train_preds and val_preds
loadpath_best = model_loads[best_train_index]
print("Reloading best training model")
model = create_model(loadweight=loadpath_best)

y_pred_train = model.predict_generator(training_generator_predict)
train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
f_train,pr_train, re_train = calc_metrics(train_preds, y_train)

saved_train_preds = pd.DataFrame(train_preds)
saved_train_preds.to_csv(savemodelpath+label+'saved_train_preds.txt',sep=',',encoding='utf-8')


loadpath_best = model_loads[best_val_index]
model = create_model(loadweight=loadpath_best)

y_pred_val = model.predict_generator(validation_generator_predict)
val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
yy = len(val_preds)
f_val,pr_val, re_val = calc_metrics(val_preds, y_val[0:yy])

saved_val_preds = pd.DataFrame(val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_val_preds.txt',sep=',',encoding='utf-8')

bootstrap_val = TP_FP_FN_forBootstrap(val_preds,y_val[0:yy])
bootstrap_val.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_val.txt',sep=',',encoding='utf-8')

##################
''' Get output '''
##################


#### PLOT USES F1_SC since this is the correct Fscore
plt.plot(range(0,epoch+1), f1_sc, color='blue', marker='o', markersize=5)
plt.plot(range(0,epoch+1), f1_sc_val, color='green', marker='s', markersize=5, linestyle ='--')
plt.plot(0, f1_sc[0], color='red', marker='o', markersize=5)
plt.plot(0, f1_sc[0], color='red', marker='x', markersize=10)
plt.annotate('Always guessing class 1', xy=(0, f1_sc[0]),xytext=(0.2,f1_sc[0]))
plt.plot(best_train_index+1, f1_sc[best_train_index+1], color='orange', marker='o', markersize=6)
plt.plot(best_val_index+1, f1_sc_val[best_val_index+1], color='orange', marker='o', markersize=6)
plt.annotate('Always guessing class 1', xy=(0, f1_sc_val[0]), xytext=(0.2,f1_sc_val[0]))

plt.plot(0, f1_sc[0], color='red', marker='o', markersize=5)
plt.plot(0, f1_sc_val[0], color='red', marker='o', markersize=5)
plt.title('model f1')
plt.ylabel('f1')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower right')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_f1.png')
plt.close()

plt.plot(range(1,epoch+1), loss, color='blue', marker='o', markersize=5)
plt.plot(range(1,epoch+1), val_loss, color='green', marker='s', markersize=5, linestyle ='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='upper right')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_loss.png')
plt.close()


plt.plot(range(1,epoch+1), precision_train, color='blue', marker='s', markersize=5)
plt.plot(range(1,epoch+1), precision_val, color='green', marker='s', markersize=5, linestyle ='--')
plt.plot(best_train_index+1, precision_train[best_train_index], color='orange', marker='o', markersize=5)
plt.plot(best_val_index+1, precision_val[best_val_index], color='orange', marker='o', markersize=5)
plt.title('Model')
plt.ylabel('Precision')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train_precision','validation_precision'], loc='upper left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_precision.png')
plt.close()


plt.plot(range(1,epoch+1), recall_train, color='blue', marker='o', markersize=5, linestyle =':')
plt.plot(range(1,epoch+1), recall_val, color='green', marker='s', markersize=5, linestyle ='-')
plt.plot(best_train_index+1, recall_train[best_train_index], color='orange', marker='o', markersize=5)
plt.plot(best_val_index+1, recall_val[best_val_index], color='orange', marker='o', markersize=5)
plt.title('Model')
plt.ylabel('Recall')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train_recall','val_recall'], loc='upper left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_recall.png')
plt.close()


print("")
print("Baseline train:", baseline_trains)
print("Baseline test:", baseline_test)
print("Baseline val:", baseline_vals)
print("")
print("Epochs:",epoch)
print("Label:",label)
print("Batch %s, Dropout of CNNs %s, Regularization %s, Dense Size %s, Drpout at Combined %s" % (batch,drop_cnn,regularization,dense_size,drop_end))

print("")
print("                         Precision, Recall,  Fscore,   tstat")
print("overall_Baseline(train): %f, %f, %f" % (total_precision_train, 1,f_baseline_train))
print("overall_Baseline(val):   %f, %f, %f" % (total_precision_val, 1,f_baseline_val))
print("overall_Baseline(test):  %f, %f, %f" % (total_precision_test, 1,f_baseline_test))
print("")
print("")
print("(train):                 %f, %f, %f" % (precision_train[best_train_index], recall_train[best_train_index],f1_sc[best_train_index+1]))
print("(val):                   %f, %f, %f" % (precision_val[best_val_index],recall_val[best_val_index],f1_sc_val[best_val_index+1]))
print("")
print("(test):                  %f, %f, %f" % (pr_test,re_test,f_test))
print("")
print("")
print("")
print(">>      Copy in txt format!")




###########################################################
''' EnvPerformance + Text --> Financial (Combined Path) '''
###########################################################

# new genertor for new model, as data input differs
class multi_input_generator_env(Sequence):
    ''' 
    Generates data for Keras
    
    list_IDs =  a list of npy. files to load
    labels   =  a dictionary of labels {'filename1.npy':1,'filename1.npy':0,...etc}
    filepath =  for example '.../testing_generator/'
    '''
    
    def __init__(self, list_IDs, env, labels, filepath, batch_size=32, sentence_length=1000, features=768, ind_dim=8, shuffle=True, to_fit=True):
        ''' initialization '''
        self.list_IDs = list_IDs
        self.env = env
        self.labels = labels
        self.batch_size = batch_size
        self.sentence_length = sentence_length 
        self.features = features
        self.ind_dim = ind_dim
        self.filepath = filepath
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()
        
    def __len__(self):
        ''' Denotes the number of batches per epoch '''
        
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        ''' 
        Generate one batch of data
        :param index: index of the batch; is created when called!
        :return: X and y when fitting. X only when predicting
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._generate_data(list_IDs_temp)
        

        if self.to_fit:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # suffles list IN PLACE! so does NOT create new list 
            
            
    def _generate_data(self, list_IDs_temp):
        '''
        Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        
        list_IDs_temp is created when __getitem__ is called
        
        '''
        # Initialization
        X = np.empty((self.batch_size, self.sentence_length, self.features))
        y = np.empty((self.batch_size), dtype=int)
        
        environmental = np.empty((self.batch_size), dtype=int)
        
        for i, ID in enumerate(list_IDs_temp):
            # i is a number;
            # ID is the file-name
               
            # load single file
            single_file = np.load(os.path.join(self.filepath,ID))
            ## create empty array to contain batch of features and labels
            batch_features = np.zeros((self.sentence_length, self.features))
            
            #####
            # to allow for shorter than 1000-sentences
            single_file = single_file[:self.sentence_length,:self.features]
            
            #####
            
            # pad loaded array to same length        
            shape = np.shape(single_file)
            batch_features[:shape[0],:shape[1]] = single_file 
            
            ## append to sequence
            X[i,] = batch_features
            
            environmental[i] = self.env[ID]
            
            y[i] = self.labels[ID] ### this looks-up according rating (Note ID = file name.npy)
            
        return [X, environmental], y



####################################
''' set labels and hyperparameter'''
####################################

label = 'EPS_q'    ## << HERE 'EPS_q', 'BHAR_z'

y_train = y_train_eps ## << HERE  y_train_eps, y_train_bhar
y_test = y_test_eps   ## << HERE  y_test_eps, y_test_bhar
y_val = y_val_eps     ## << HERE  y_val_eps, y_val_bhar

gold_train = y_train_ep90 ## << HERE  y_train_EP, y_train_ep90, 
gold_test = y_test_ep90   ## << HERE y_test_EP, y_test_ep90, 
gold_val = y_val_ep90     ## << HERE y_val_EP, y_val_ep90


label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)

epoch = 10
batch = 32
drop_cnn = 0.5
regularization = 0.01  ## check if l1, l1_l2 or just l2 regularization!
dense_size = 50
drop_end = 0.5

no_sentences_per_doc = 1000 #1000
sentence_embedding = 768  # if just normal seq
#X, y, y_price = feature_files_pd['np_name'].to_list(), feature_files_pd[labelname].to_list(), feature_files_pd[secondlabel].to_list()
# Create a zip object from two lists

print("For", label)
print("Test size %s, Train size %s, Val size %s" % (len(y_test),len(y_train),len(y_val)))
y_train, y_test, y_val = np.array(y_train, dtype='float32'), np.array(y_test, dtype='float32'), np.array(y_val, dtype='float32')

X_train, X_test, X_val = np.array(x_train, dtype='str'), np.array(x_test, dtype='str'), np.array(x_val, dtype='str')

y_label_train = dict(zip(X_train,y_train)) # labels is actual input for generator
y_label_val = dict(zip(X_val,y_val)) # labels is actual input for generator
y_label_test = dict(zip(X_test,y_test)) # labels is actual input for generator


env_golden_train = dict(zip(X_train,gold_train)) # labels is actual input for generator
env_golden_test = dict(zip(X_test,gold_test)) # labels is actual input for generator
env_golden_val = dict(zip(X_val,gold_val)) # labels is actual input for generator



def create_model(loadweight=None):
    ##### MODEL #####
    sequence_input  = Input(shape=(no_sentences_per_doc, sentence_embedding))

    ### environmental performance
    env_perf = Input(shape=(1,), dtype='float32')
    
    gru_layer = Bidirectional(GRU(50, #activation='tanh',
                              return_sequences=True#True
                              ))(sequence_input)
    
    sent_dense = Dense(100, activation='relu', name='sent_dense')(gru_layer)  # make signal stronger
    
    sent_att,sent_coeffs = AttentionLayer(100,return_coefficients=True,name='sent_attention')(sent_dense)
    sent_att = Dropout(0.5,name='sent_dropout')(sent_att)

    ## combine market cap and industry with cnn
    combined = Concatenate()([sent_att, env_perf])


    preds = Dense(1, activation='sigmoid',name='output')(combined)  # NOTE: SIGMOID FOR 2-CLASS PROBLEM; softmax for multiclass problem
    model = Model([sequence_input,env_perf], preds)

    if loadweight is not None:
        model.load_weights(loadweight)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[TruePositives(name='true_positives'),
                          TrueNegatives(name='true_negatives'),
                          FalseNegatives(name='false_negatives'),
                          FalsePositives(name='false_positives'),
                          #f1
                          ])
    return model


##########################
''' Calculate Baseline '''
##########################

unique, counts = np.unique(y_train,                 
                           return_counts=True)
baseline_trains = dict(zip(unique, counts))
total = baseline_trains[1] + baseline_trains[0]
total_precision_train = baseline_trains[1] / total
f_baseline_train = 2*(total_precision_train)/(1+total_precision_train)
print("Baseline_train:",baseline_trains)

unique_val, counts_val = np.unique(y_val,
                                   return_counts=True)
baseline_vals = dict(zip(unique_val, counts_val))
total_val = baseline_vals[1] + baseline_vals[0]
total_precision_val = baseline_vals[1] / total_val
f_baseline_val = 2*(total_precision_val)/(1+total_precision_val)
print("Baseline_val:",baseline_vals)

unique_test, counts_test = np.unique(y_test,
                                     return_counts=True)
baseline_test = dict(zip(unique_test, counts_test))
total_test = baseline_test[1] + baseline_test[0]
total_precision_test = baseline_test[1] / total_test
f_baseline_test = 2*(total_precision_test)/(1+total_precision_test)
print("Baseline_test:",baseline_test)


#################
''' run model '''
#################

accuracy, val_accuracy, manual_acc = [], [], []
loss, val_loss = [], []
fscores, f_m, f_m_val = [], [], []

precision_sc, precision_sc_val = [], []
recall_sc, recall_sc_val = [], []

f1_sc, f1_sc_val = [], []

tp, fp, fn = [], [], []
tp_val, fp_val, fn_val = [], [], []

# other variable that must be fed is a dictionary of labels
training_generator = multi_input_generator_env(X_train, env_golden_train, y_label_train,
                                               bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                               features=sentence_embedding, shuffle=False, to_fit=True)
validation_generator = multi_input_generator_env(X_val, env_golden_val, y_label_val,
                                                 bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                 features=sentence_embedding, shuffle=False, to_fit=True)  # to_fit returns X and y

model = create_model()

# checkpoint
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), verbose=1, save_best_only=False, save_weights_only=True, mode='auto')#, save_freq='epoch')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator, ## << HERE
                              epochs=epoch,
                              callbacks=callbacks_list
                              )


loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])

loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()

######################
''' Reloade models '''
######################

model_loads = []
for root, dirs, files in os.walk(savemodelpath):
    for file in files:
        if file.endswith(".hdf5"):  
            model_loads.append(os.path.join(root,file))

Fscore_val, Fscore_train = [], []

fscorepredict_val, fscorepredict_train = [], []
fscorepredict_val_custom, fscorepredict_val_sklearn = [], []

recall_train, recall_val = [], []
precision_train, precision_val = [], []

training_generator_predict = multi_input_generator_env(X_train, env_golden_train, y_label_train,
                                                       bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                       features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y

validation_generator_predict = multi_input_generator_env(X_val, env_golden_val, y_label_val,
                                                         bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                         features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y

cnt = 0
for i in model_loads:  
    percent = cnt / epoch
    print("Progress: ",percent)
    ## recreating model
    model = create_model(loadweight=i)

    # for validation data
    y_pred = model.predict_generator(validation_generator_predict, verbose=0)
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    
    ## see Careful-note
    yy = len(val_preds)
    cm, p, r = calc_metrics(val_preds, y_val[0:yy])#y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    
    cm = f1_score(y_val[0:yy], val_preds)
    #cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)

    # for training data
    y_pred =  model.predict_generator(training_generator_predict, verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)
    cnt += 1


# saving loss and fscore for plots
diction = {'loss':loss,'val_loss':val_loss,'fscore':fscorepredict_train,'val_fscore':fscorepredict_val_custom}
df = pd.DataFrame(diction, columns=['loss','val_loss','fscore','val_fscore'])
df.to_csv(savemodelpath+label+'loss_fscore.txt', sep=',', encoding='utf-8')

diction = {'pr_train':precision_train,'pr_val':precision_val,'rec_train':recall_train,'rec_val':recall_val}
df2 = pd.DataFrame(diction, columns=['pr_train','pr_val','rec_train','rec_val'])
df2.to_csv(savemodelpath+label+'recall_precision.txt', sep=',', encoding='utf-8')


fscorepredict_train = np.array(fscorepredict_train)
fscorepredict_val_custom = np.array(fscorepredict_val_custom)

best_validation_model = np.argmax(fscorepredict_val_custom)

best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)

f1_sc = np.insert(fscorepredict_train, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(fscorepredict_val_custom, 0, f_baseline_val, axis=0)


test_generator_predict = multi_input_generator_env(X_test, env_golden_test, y_label_test,
                                                   bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                   features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y


# run best model (highest validation score)
print("Reloading best validation model")
loadpath_best = model_loads[best_validation_model]
model = create_model(loadweight=loadpath_best)

y_pred = model.predict_generator(test_generator_predict)

test_preds = [1 if x > 0.5 else 0 for x in y_pred]
yy = len(test_preds)

f_test,pr_test, re_test = calc_metrics(test_preds, y_test[0:yy])

bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test[0:yy])
bootstrap_test.to_csv(savemodelpath+label+'bootstrap_test.txt',sep=',',encoding='utf-8')

# saving best train_preds and val_preds
loadpath_best = model_loads[best_train_index]
print("Reloading best training model")
model = create_model(loadweight=loadpath_best)

y_pred_train = model.predict_generator(training_generator_predict)
train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
f_train,pr_train, re_train = calc_metrics(train_preds, y_train)

saved_train_preds = pd.DataFrame(train_preds)
saved_train_preds.to_csv(savemodelpath+label+'saved_train_preds.txt',sep=',',encoding='utf-8')

loadpath_best = model_loads[best_val_index]
model = create_model(loadweight=loadpath_best)

y_pred_val = model.predict_generator(validation_generator_predict)
val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
yy = len(val_preds)
f_val,pr_val, re_val = calc_metrics(val_preds, y_val[0:yy])

saved_val_preds = pd.DataFrame(val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_plus_env_saved_val_preds.txt',sep=',',encoding='utf-8')

bootstrap_val = TP_FP_FN_forBootstrap(val_preds,y_val[0:yy])
bootstrap_val.to_csv(savemodelpath+label+'_plus_env_bootstrap_val.txt',sep=',',encoding='utf-8')

##################
''' Get Output '''
##################


print("")
print("Baseline train:", baseline_trains)
print("Baseline test:", baseline_test)
print("Baseline val:", baseline_vals)
print("")
print("Epochs:",epoch)
print("Label:",label)
print("Batch %s, Dropout of CNNs %s, Regularization %s, Dense Size %s, Drpout at Combined %s" % (batch,drop_cnn,regularization,dense_size,drop_end))

print("")
print("                         Precision, Recall,  Fscore,   tstat")
print("overall_Baseline(train): %f, %f, %f" % (total_precision_train, 1,f_baseline_train))
print("overall_Baseline(val):   %f, %f, %f" % (total_precision_val, 1,f_baseline_val))
print("overall_Baseline(test):  %f, %f, %f" % (total_precision_test, 1,f_baseline_test))
print("")
print("")
print("(train):                 %f, %f, %f" % (precision_train[best_train_index], recall_train[best_train_index],f1_sc[best_train_index+1]))
print("(val):                   %f, %f, %f" % (precision_val[best_val_index],recall_val[best_val_index],f1_sc_val[best_val_index+1]))
print("")
print("(test):                  %f, %f, %f" % (pr_test,re_test,f_test))
print("")
print("")
print("")
print(">>      Copy in txt format!")



#### PLOT USES F1_SC since this is the correct Fscore 
plt.plot(range(0,epoch+1), f1_sc, color='blue', marker='o', markersize=5)
plt.plot(range(0,epoch+1), f1_sc_val, color='green', marker='s', markersize=5, linestyle ='--')
plt.plot(0, f1_sc[0], color='red', marker='o', markersize=5)
plt.plot(0, f1_sc[0], color='red', marker='x', markersize=10)
plt.annotate('Always guessing class 1', xy=(0, f1_sc[0]),xytext=(0.2,f1_sc[0]))
plt.plot(best_train_index+1, f1_sc[best_train_index+1], color='orange', marker='o', markersize=6)
plt.plot(best_val_index+1, f1_sc_val[best_val_index+1], color='orange', marker='o', markersize=6)
plt.annotate('Always guessing class 1', xy=(0, f1_sc_val[0]), xytext=(0.2,f1_sc_val[0]))

plt.plot(0, f1_sc[0], color='red', marker='o', markersize=5)
plt.plot(0, f1_sc_val[0], color='red', marker='o', markersize=5)
plt.title('model f1')
plt.ylabel('f1')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower right')
#plt.show()
plt.savefig(savemodelpath+label+'_f1.png')
plt.close()

plt.plot(range(1,epoch+1), loss, color='blue', marker='o', markersize=5)
plt.plot(range(1,epoch+1), val_loss, color='green', marker='s', markersize=5, linestyle ='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='upper right')
#plt.show()
plt.savefig(savemodelpath+label+'_loss.png')
plt.close()


plt.plot(range(1,epoch+1), precision_train, color='blue', marker='s', markersize=5)
plt.plot(range(1,epoch+1), precision_val, color='green', marker='s', markersize=5, linestyle ='--')
plt.plot(best_train_index+1, precision_train[best_train_index], color='orange', marker='o', markersize=5)
plt.plot(best_val_index+1, precision_val[best_val_index], color='orange', marker='o', markersize=5)
plt.title('Model')
plt.ylabel('Precision')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train_precision','validation_precision'], loc='upper left')
#plt.show()
plt.savefig(savemodelpath+label+'_precision.png')
plt.close()


plt.plot(range(1,epoch+1), recall_train, color='blue', marker='o', markersize=5, linestyle =':')
plt.plot(range(1,epoch+1), recall_val, color='green', marker='s', markersize=5, linestyle ='-')
plt.plot(best_train_index+1, recall_train[best_train_index], color='orange', marker='o', markersize=5)
plt.plot(best_val_index+1, recall_val[best_val_index], color='orange', marker='o', markersize=5)
plt.title('Model')
plt.ylabel('Recall')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train_recall','val_recall'], loc='upper left')
#plt.show()
plt.savefig(savemodelpath+label+'_recall.png')
plt.close()

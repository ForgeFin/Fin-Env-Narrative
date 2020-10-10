
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
''' Define custom metrics for calculating F-Score, precision, and recall '''
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
            


#############################################################################
''' Define your directory containing all documents'''
#############################################################################

bertdocs = '.../Bert_Docs/' ## directory containing the txt-files 
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

## LOAD dataframe containing the label data for all documents
## format: filename (fname), labels (Env_q, Env_t, EPS_q, BHAR_t), market capitalization (Log_MarketCap), and industry identifier (osha_gov_SIC_numeric)
data_all_rank = pd.read_csv(save_path+'/dataframe.txt', sep=',', index_col=0, encoding='utf-8')

random = 0 
labelname = 'env_t'     # or 'env_q'
secondlabel = 'BHAR_t'  #'EPS_q'
X, y, y_price = data_all_rank['file_loads_one_sent'].to_list(), data_all_rank[labelname].to_list(), data_all_rank[secondlabel].to_list()


# train, test, val split
X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.1, random_state=random)       ## first train,test,val random_state=1
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random)

y_train, y_test, y_val = np.array(y_train, dtype='float32'), np.array(y_test, dtype='float32'), np.array(y_val, dtype='float32')
print("Test size %s, Train size %s, Val size %s" % (len(y_test),len(y_train),len(y_val)))


#############################################################################
''' Create dataframes (train, test, val) with file-location, labels,
market capitalization and instustry identifier'''
#############################################################################

save_X_train = pd.DataFrame(X_train)
save_X_train[['path','train']] = save_X_train[0].str.split(bertdocs, n = 1, expand = True) 
save_X_train = save_X_train.loc[:, ['train']]
train = save_X_train['train'].tolist()
traindict = dict(zip(train,y_train))
train = pd.DataFrame.from_dict(traindict, orient='index',columns=['BHAR_t'])
train.index.name = 'fname'
train = train.reset_index(level='fname')

save_X_test = pd.DataFrame(X_test)
save_X_test[['path','test']] = save_X_test[0].str.split(bertdocs, n = 1, expand = True) 
save_X_test = save_X_test.loc[:, ['test']]
test = save_X_test['test'].tolist()
testdict = dict(zip(test,y_test))
test = pd.DataFrame.from_dict(testdict, orient='index',columns=['BHAR_t'])
test.index.name = 'fname'
test = test.reset_index(level='fname')

save_X_val = pd.DataFrame(X_val)
save_X_val[['path','val']] = save_X_val[0].str.split(bertdocs, n = 1, expand = True) 
save_X_val = save_X_val.loc[:, ['val']]
val = save_X_val['val'].tolist()
valdict = dict(zip(val,y_val))
val = pd.DataFrame.from_dict(valdict, orient='index',columns=['BHAR_t'])
val.index.name = 'fname'
val = val.reset_index(level='fname')


validation_all = val.merge(data_all_rank[['fname','env_t','env_q','EPS_q']], on=['fname'], how='inner')
train_all = train.merge(data_all_rank[['fname','env_t','env_q','EPS_q']], on=['fname'], how='inner')
test_all = test.merge(data_all_rank[['fname','env_t','env_q','EPS_q']], on=['fname'], how='inner')
validation_all.to_csv(saver+'validation_data_rs'+str(random)+'.txt', sep=',', encoding='utf-8')
train_all.to_csv(saver+'train_data_rs'+str(random)+'.txt', sep=',', encoding='utf-8')
test_all.to_csv(saver+'test_data_rs'+str(random)+'.txt', sep=',', encoding='utf-8')

## Reloading
#validation_all = pd.read_csv(saver+'validation_data_rs'+str(random)+'.txt', sep=',', index_col=0, encoding='utf-8')
#train_all = pd.read_csv(saver+'train_data_rs'+str(random)+'.txt', sep=',', index_col=0, encoding='utf-8')
#test_all = pd.read_csv(saver+'test_data_rs'+str(random)+'.txt', sep=',', index_col=0, encoding='utf-8')

validation_all['path'] = bertdocs
train_all['path'] = bertdocs
test_all['path'] = bertdocs

validation_all['fileload'] = validation_all[['path', 'fname']].apply(lambda x: ''.join(x), axis=1)
train_all['fileload'] = train_all[['path', 'fname']].apply(lambda x: ''.join(x), axis=1)
test_all['fileload'] = test_all[['path', 'fname']].apply(lambda x: ''.join(x), axis=1)


## Adding market cap and industry
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

## Note: Different labels are assigned here
x_test, y_test_EP, y_test_bhar, y_test_ep90, y_test_eps = test_all['fileload'], test_all['env_t'], test_all['BHAR_t'], test_all['env_q'], test_all['EPS_q']
x_train, y_train_EP, y_train_bhar, y_train_ep90, y_train_eps = train_all['fileload'], train_all['env_t'], train_all['BHAR_t'], train_all['env_q'], train_all['EPS_q']
x_val, y_val_EP, y_val_bhar, y_val_ep90, y_val_eps = validation_all['fileload'], validation_all['env_t'], validation_all['BHAR_t'], validation_all['env_q'], validation_all['EPS_q']



#############################################################################
''' Load each document, clean files for stopwords, non-alphanumeric tokens,
and words shorter than 2 characters '''
#############################################################################

print("Loading text to memory!")


first_test = len(x_test)
second_train = len(x_train)
third_val = len(x_val)

## append all document IDs to one list
x_test = list(x_test)
x_test.extend(list(x_train))
x_test.extend(list(x_val))
print("All files:", len(x_test))

def cleanstr(file_list):
    text = []
    translator = str.maketrans(punctuation, ' '*len(punctuation)) ## splits thi-s as "thi" "s"
    for f in file_list:
        with open(f, encoding='utf-8') as document:
            doc = document.read()
            doc = doc.lower()
            tok = doc.translate(translator)
            tokens = tok.split()

            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
            
            # filter short tokens
            tokens = [word for word in tokens if len(word) > 2]  #### changed to 2
            docs = " ".join(tokens)
            text.append(docs)
    return text


##################################################
''' Apply keras tokenizer and sequence padding '''
##################################################


text = cleanstr(x_test)
#MAX_FEATURES = 200000 # set the maximum number of words to keep equal to 200,000

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(text)

print("Processed documents", tokenizer.document_count) # int. Number of documents (texts/sequences) the tokenizer was trained on

# tokenizer just tokenizes but need to be in sequence (vector) format 
sequences = tokenizer.texts_to_sequences(text)

# maxlen: Int, maximum length of all sequences.
MAX_SEQUENCE_LENGTH = 20000 ## Note documents with more words than max_seq are cuttet!
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

## recreate x_test, x_train, x_val from processed data
x_test = data[:first_test]
print("first test length", first_test)
print("x_test,",len(x_test))
print("")
scnd = first_test + second_train
x_train = data[first_test:scnd]
print("second_train length:", second_train)
print("x_train",len(x_train))
lastone = first_test+second_train
x_val = data[lastone::]
print("x_test",len(x_val))


############################################
''' Create glove-embedding layer (100dim)'''
############################################

# Note: Choose your path containing the glove-embeddings
# or use https://nlp.stanford.edu/projects/glove/
glove_path = '.../glove.6B/'


# Load word vectors from pre-trained dataset
embeddings_index = {}
f = open(glove_path+'glove.6B.100d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Embedding
EMBED_SIZE = 100
min_wordCount = 2
absent_words = 0
small_words = 0

word_index = tokenizer.word_index 
print('Found %s unique tokens.' % len(word_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
word_counts = tokenizer.word_counts
for word, i in word_index.items():
    if word_counts[word] > min_wordCount:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            absent_words += 1
    else:
        small_words += 1
        
print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)),
      '% of total words') ## absent words = words not in GloVe embedding matrix
print('Words with '+str(min_wordCount)+' or less mentions', small_words, 'which is', "%0.2f" % (small_words * 100 / len(word_index)),
      '% of total words')
## only according number of words to proceed are accounted for; other words do not exist as a word vector OR occurance is to small
print(str(len(word_index)-small_words-absent_words) + ' words to proceed.')


embedding_layer = Embedding(len(word_index) + 1,
                            EMBED_SIZE,
                            weights=[embedding_matrix], # by choosing embedding matrix only 27037 words are used
                            input_length=MAX_SEQUENCE_LENGTH, # input_length: Length of input sequences, when it is constant
                            trainable=False) # trainable is False, so is not updated while training



#####################################################
''' Model setup for financial label (Direct Path) '''
#####################################################

# Setting y
label = 'EPS_q'          ## << HERE 'EPS_q', 'BHAR_t'
y_train = y_train_eps    ## << HERE  y_train_eps, y_train_bhar
y_test = y_test_eps      ## << HERE  y_test_eps, y_test_bhar
y_val = y_val_eps        ## << HERE y_val_eps, y_val_bhar


label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)


epoch = 50
batch = 32
drop_cnn = 0.5
regularization = 0.01
dense_size = 50
drop_end = 0.5
input_dim = x_train.shape[1] #1000
num_workers = cpu_count()

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


def create_model(loadweight=None):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    conv1 = Conv1D(filters=2, kernel_size=2, padding='same')(embedded_sequences)
    conv1 = MaxPooling1D(pool_size=32)(conv1)
    conv1 = Dropout(drop_cnn)(conv1)
    
    conv2 = Conv1D(filters=2, kernel_size=3, padding='same')(embedded_sequences)
    conv2 = MaxPooling1D(pool_size=32)(conv2)
    conv2 = Dropout(drop_cnn)(conv2)
    
    conv3 = Conv1D(filters=2, kernel_size=4, padding='same')(embedded_sequences)
    conv3 = MaxPooling1D(pool_size=32)(conv3)
    conv3 = Dropout(drop_cnn)(conv1)
    
    cnn = Concatenate(axis=-1)([conv1, conv2, conv3])
    flat = Flatten()(cnn)

    x = Dense(dense_size, activation="relu",
              kernel_regularizer=regularizers.l2(regularization),          ## l2\n",
              bias_regularizer=regularizers.l2(regularization))(flat)      ## l2\n",
    x = Dropout(drop_end)(x)
    
    preds = Dense(1, activation='sigmoid',name='output')(x)  # NOTE: SIGMOID FOR 2-CLASS PROBLEM; softmax for multiclass problem\n",
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


############################
''' Baseline calculation '''
############################

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

model = create_model()

# Save each model after epoch; save checkpoint
filepath="weights-improvement-{epoch:02d}-{F1_score_val:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), monitor='F1_score_val', verbose=1, save_best_only=False, save_weights_only=True, mode='auto')#, save_freq='epoch')
#callbacks_list = [checkpoint]

history = model.fit(x_train, y_train,                
                    validation_data=(x_val, y_val),
                    epochs=epoch, batch_size=batch,
                    callbacks=[F1History(train=(x_train,y_train),validation=(x_val,y_val)),checkpoint]
                    )


#############################
''' Save output and plots '''
#############################

loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])

f1_sc.append(history.history['F1_score_train'])
f1_sc_val.append(history.history['F1_score_val'])

f1_sc = np.array(f1_sc).flatten()
f1_sc_val = np.array(f1_sc_val).flatten()


best_validation_model = np.argmax(f1_sc_val)

best_trainings_fscore = f1_sc[np.argsort(f1_sc)[-1:]]
best_validation_fscore = f1_sc_val[np.argsort(f1_sc_val)[-1:]]
best_val_index = np.argmax(f1_sc_val)
best_train_index = np.argmax(f1_sc)

f1_sc = np.insert(f1_sc, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(f1_sc_val, 0, f_baseline_val, axis=0)


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
plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower right')
#plt.show()
plt.savefig(savemodelpath+label+'_f1.png')
plt.close()

loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()


plt.plot(range(1,epoch+1), loss, color='blue', marker='o', markersize=5)
plt.plot(range(1,epoch+1), val_loss, color='green', marker='s', markersize=5, linestyle ='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower left')
#plt.show()
plt.savefig(savemodelpath+label+'_loss.png')
plt.close()



##########################################
''' Reload models and calculate Fscore '''
##########################################

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

cnt = 0
for i in model_loads: 
    ## Note: Calculates Fscore using custom metric and fscore from sklearn
    model = create_model(loadweight=i)
    
    # for training data
    y_pred =  model.evaluate(x_train, y_train, verbose=0)  # gives probabilities
    Fscore_train.append(y_pred)
    
    y_pred_train =  model.predict(x_train, verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
    if cnt == best_train_index:
        saved_train_preds = train_preds
    
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)
    
    
    # for validation data
    y_prob =  model.evaluate(x_val, y_val, verbose=0)  # gives probabilities
    Fscore_val.append(y_prob)
    
    y_pred_val = model.predict(x_val)
    val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
    if cnt == best_val_index:
        saved_val_preds = val_preds
    
    cm, p, r = calc_metrics(val_preds, y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)
    cnt += 1


## save bootstraps for best validation models
bootstrap_val = TP_FP_FN_forBootstrap(saved_val_preds,y_val)
bootstrap_val.to_csv(savemodelpath+label+'_bootstrap_val.txt',sep=',',encoding='utf-8')

saved_val_preds = pd.DataFrame(saved_val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_saved_val_preds.txt',sep=',',encoding='utf-8')
saved_train_preds = pd.DataFrame(saved_train_preds)
saved_train_preds.to_csv(savemodelpath+label+'saved_train_preds.txt',sep=',',encoding='utf-8')


## run best model (highest validation score)
loadpath_best = model_loads[best_validation_model]

model = create_model(loadweight=loadpath_best)

## Compile model (required to make predictions)
model.compile(loss='binary_crossentropy',
          optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
                   ])

y_pred_test = model.predict(x_test)

test_preds = [1 if x > 0.5 else 0 for x in y_pred_test]
f_test,pr_test, re_test = calc_metrics(test_preds, y_test)

bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test)
bootstrap_test.to_csv(savemodelpath+label+'_bootstrap_test.txt',sep=',',encoding='utf-8')


##############################
''' Print and plot results '''
##############################

print("")
print("Baseline train:", baseline_trains)
print("Baseline test:", baseline_test)
print("Baseline val:", baseline_vals)
print("")
print("Batch %s, Dropout of CNNs %s, Regularization %s, Dense Size %s, Drpout at Combined %s" % (batch,drop_cnn,regularization,dense_size,drop_end))
print("Epochs:",epoch)
print("Label:",label)
print("")
print("                         Precision, Recall,  Fscore")
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


plt.plot(range(1,epoch+1), precision_train, color='blue', marker='s', markersize=5)
plt.plot(range(1,epoch+1), precision_val, color='green', marker='s', markersize=5, linestyle ='--')
plt.plot(best_train_index+1, precision_train[best_train_index], color='orange', marker='o', markersize=5)
plt.plot(best_val_index+1, precision_val[best_val_index], color='orange', marker='o', markersize=5)
plt.title('Model')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.xticks(range(0, epoch+1))
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
plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train_recall','val_recall'], loc='upper left')
#plt.show()
plt.savefig(savemodelpath+label+'_recall.png')
plt.close()

print("")
print(">>      Copy in txt format!")




##########################################################################################
''' Model setup for environmental label considering Market Cap and Industry (path env) '''
##########################################################################################


label = 'env_t'                   ## << HERE 'env_t','env_q'
y_train = np.array(y_train_EP)    ## << HERE  y_train_EP, y_train_ep90
y_test = np.array(y_test_EP)      ## << HERE  y_test_EP, y_test_ep90
y_val = np.array(y_val_EP)        ## << HERE  y_val_EP, y_val_ep90


label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)


epoch = 50
batch = 32
drop_cnn = 0.5
regularization = 0.01
dense_size = 50
drop_end = 0.5
industry_dim = industry_train.shape[1]

def create_model(loadweight=None):
    #print('Testing %s label..' % labelname)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    ### ADDING MARKET CAP AND INDUSTRY
    market_cap = Input(shape=(1,), dtype='float32')
    industry = Input(shape=(industry_dim,), dtype='float32')

    conv1 = Conv1D(filters=2, kernel_size=2, padding='same')(embedded_sequences)
    conv1 = MaxPooling1D(pool_size=32)(conv1)
    conv1 = Dropout(drop_cnn)(conv1)

    conv2 = Conv1D(filters=2, kernel_size=3, padding='same')(embedded_sequences)
    conv2 = MaxPooling1D(pool_size=32)(conv2)
    conv2 = Dropout(drop_cnn)(conv2)

    conv3 = Conv1D(filters=2, kernel_size=4, padding='same')(embedded_sequences)
    conv3 = MaxPooling1D(pool_size=32)(conv3)
    conv3 = Dropout(drop_cnn)(conv3)

    cnn = Concatenate(axis=-1)([conv1, conv2, conv3])
    flat = Flatten()(cnn)

    x = Dense(dense_size, activation="relu",
              kernel_regularizer=regularizers.l2(regularization),
              bias_regularizer=regularizers.l2(regularization))(flat) ## combined 

    ## combine market cap and industry with cnn
    combined = Concatenate()([x, market_cap, industry])

    x = Dropout(drop_end)(combined)
        
    preds = Dense(1, activation='sigmoid',name='output')(x)  # NOTE: SIGMOID FOR 2-CLASS PROBLEM; softmax for multiclass problem
    model = Model([sequence_input,market_cap,industry], preds)

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


############################
''' Baseline calculation '''
############################

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

model = create_model()

# save checkpoints
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

history = model.fit([x_train,mktcap_train,industry_train], y_train,               
                    validation_data=([x_val,mktcap_val,industry_val], y_val),
                    epochs=epoch, batch_size=batch,
                    callbacks=[checkpoint]
                    )

loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])

loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()


##########################################
''' Reload models and calculate Fscore '''
##########################################

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


for i in model_loads:  
    ## recreating model
    model = create_model(loadweight=i)

    # for validation data
    y_pred =  model.evaluate([x_val,mktcap_val,industry_val
                              ], y_val, verbose=0)  # gives probabilities
    Fscore_val.append(y_pred)
    
    y_pred = model.predict([x_val,mktcap_val,industry_val
                            ])
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p, r = calc_metrics(val_preds, y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)


    # for training data
    y_pred =  model.evaluate([x_train,mktcap_train,industry_train
                              ], y_train, verbose=0)  # gives probabilities
    Fscore_train.append(y_pred)
    
    y_pred =  model.predict([x_train,mktcap_train,industry_train
                             ], verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)


fscorepredict_train = np.array(fscorepredict_train)
fscorepredict_val_custom = np.array(fscorepredict_val_custom)

best_validation_model = np.argmax(fscorepredict_val_custom)

best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)

f1_sc = np.insert(fscorepredict_train, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(fscorepredict_val_custom, 0, f_baseline_val, axis=0)

# running best model (highest validation score)
loadpath_best = model_loads[best_validation_model]

## recreating model
model = create_model(loadweight=loadpath_best)

y_pred = model.predict([x_test,mktcap_test,industry_test
                        ])

test_preds = [1 if x > 0.5 else 0 for x in y_pred]
f_test,pr_test, re_test = calc_metrics(test_preds, y_test)

bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test)
bootstrap_test.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_test.txt',sep=',',encoding='utf-8')

# saving best train_preds and val_preds
loadpath_best = model_loads[best_train_index]

model = create_model(loadweight=loadpath_best)

y_pred_train = model.predict([x_train,mktcap_train,industry_train])
train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
f_train,pr_train, re_train = calc_metrics(train_preds, y_train)

saved_train_preds = pd.DataFrame(train_preds)
saved_train_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_train_preds.txt',sep=',',encoding='utf-8')


loadpath_best = model_loads[best_val_index]
model = create_model(loadweight=loadpath_best)

y_pred_val = model.predict([x_val,mktcap_val,industry_val])
val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
f_val,pr_val, re_val = calc_metrics(val_preds, y_val)

saved_val_preds = pd.DataFrame(val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_val_preds.txt',sep=',',encoding='utf-8')

bootstrap_val = TP_FP_FN_forBootstrap(val_preds,y_val)
bootstrap_val.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_val.txt',sep=',',encoding='utf-8')



##############################
''' Print and plot results '''
##############################

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
plt.xticks(range(0, epoch+1))
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




#################################################################################################
''' Model setup for Environmental Performance + Text ->  Financial Performance (Combined path)'''
#################################################################################################

label = 'EPS_q'          ## << HERE 'EPS_q', 'BHAR_t'
y_train = y_train_eps    ## << HERE  y_train_eps, y_train_bhar
y_test = y_test_eps      ## << HERE  y_test_eps, y_test_bhar
y_val = y_val_eps        ## << HERE y_val_eps, y_val_bhar

golden = 'env_t'                  ## << HERE 'env_t','env_q' 
golden_env_train = y_train_ep90   ## << HERE  y_train_EP, y_train_ep90, 
golden_env_test = y_test_ep90     ## << HERE  y_test_EP, y_test_ep90,
golden_env_val = y_val_ep90       ## << HERE  y_val_EP, y_val_ep90,

label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)


epoch = 50
batch = 32
input_dim = x_train.shape[1] #1000

golden_env_train = np.array(golden_env_train, dtype='float32')
golden_env_test = np.array(golden_env_test, dtype='float32')
golden_env_val = np.array(golden_env_val, dtype='float32')

num_workers = cpu_count()

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


epoch = 50
batch = 32
drop_cnn = 0.5
regularization = 0.01
dense_size = 50
drop_end = 0.5
industry_dim = industry_train.shape[1]

def create_model(loadweight=None):
    #print('Testing %s label..' % labelname)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    ### ADDING Environmental performance
    env_perf = Input(shape=(1,), dtype='float32')

    conv1 = Conv1D(filters=2, kernel_size=2, padding='same')(embedded_sequences)
    conv1 = MaxPooling1D(pool_size=32)(conv1)
    conv1 = Dropout(drop_cnn)(conv1)

    conv2 = Conv1D(filters=2, kernel_size=3, padding='same')(embedded_sequences)
    conv2 = MaxPooling1D(pool_size=32)(conv2)
    conv2 = Dropout(drop_cnn)(conv2)

    conv3 = Conv1D(filters=2, kernel_size=4, padding='same')(embedded_sequences)
    conv3 = MaxPooling1D(pool_size=32)(conv3)
    conv3 = Dropout(drop_cnn)(conv3)

    cnn = Concatenate(axis=-1)([conv1, conv2, conv3])
    flat = Flatten()(cnn)


    ## combine market cap and industry with cnn
    combined = Concatenate()([flat, env_perf])
 
    x = Dense(dense_size, activation="relu",
              kernel_regularizer=regularizers.l2(regularization),
              bias_regularizer=regularizers.l2(regularization))(combined)
    x = Dropout(drop_end)(x)
        
    preds = Dense(1, activation='sigmoid',name='output')(x)  # NOTE: SIGMOID FOR 2-CLASS PROBLEM; softmax for multiclass problem
    model = Model([sequence_input,env_perf], preds)

    if loadweight is not None:
        model.load_weights(loadweight)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', # opt
                  metrics=[TruePositives(name='true_positives'),
                          TrueNegatives(name='true_negatives'),
                          FalseNegatives(name='false_negatives'),
                          FalsePositives(name='false_positives'),
                          #f1
                          ])
    return model


############################
''' Baseline calculation '''
############################

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

model = create_model()

# Save checkpoints for reload
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

history = model.fit([x_train,golden_env_train], y_train,                
                    validation_data=([x_val,golden_env_val], y_val), 
                    epochs=epoch, batch_size=batch,
                    callbacks=[checkpoint]
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


for i in model_loads:  
    ## recreating model
    model = create_model(loadweight=i)
     
    # for validation data
    y_pred =  model.evaluate([x_val,golden_env_val
                              ], y_val, verbose=0)  # gives probabilities
    Fscore_val.append(y_pred)
    
    y_pred = model.predict([x_val,golden_env_val
                            ])
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p, r = calc_metrics(val_preds, y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)


    # for training data
    y_pred =  model.evaluate([x_train,golden_env_train
                              ], y_train, verbose=0)  # gives probabilities
    Fscore_train.append(y_pred)
    
    y_pred =  model.predict([x_train,golden_env_train
                             ], verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)

fscorepredict_train = np.array(fscorepredict_train)
fscorepredict_val_custom = np.array(fscorepredict_val_custom)


best_validation_model = np.argmax(fscorepredict_val_custom)

best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)

f1_sc = np.insert(fscorepredict_train, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(fscorepredict_val_custom, 0, f_baseline_val, axis=0)

# running best model (highest validation score)
loadpath_best = model_loads[best_validation_model]

## recreating model
model = create_model(loadweight=loadpath_best)

y_pred = model.predict([x_test,golden_env_test
                        ])

test_preds = [1 if x > 0.5 else 0 for x in y_pred]
f_test,pr_test, re_test = calc_metrics(test_preds, y_test)

bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test)
bootstrap_test.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_test.txt',sep=',',encoding='utf-8')


# saving best train_preds and val_preds
loadpath_best = model_loads[best_train_index]

model = create_model(loadweight=loadpath_best)

y_pred_train = model.predict([x_train,golden_env_train])
train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
f_train,pr_train, re_train = calc_metrics(train_preds, y_train)

saved_train_preds = pd.DataFrame(train_preds)
saved_train_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_train_preds.txt',sep=',',encoding='utf-8')

loadpath_best = model_loads[best_val_index]
model = create_model(loadweight=loadpath_best)

y_pred_val = model.predict([x_val,golden_env_val])
val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
f_val,pr_val, re_val = calc_metrics(val_preds, y_val)

saved_val_preds = pd.DataFrame(val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_val_preds.txt',sep=',',encoding='utf-8')

bootstrap_val = TP_FP_FN_forBootstrap(val_preds,y_val)
bootstrap_val.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_val.txt',sep=',',encoding='utf-8')


##############################
''' Print and plot results '''
##############################

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
plt.xticks(range(0, epoch+1))
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







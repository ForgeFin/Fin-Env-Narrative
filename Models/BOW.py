

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
data_all_rank = pd.read_csv(save_path+'/2020_15_07_all_samples_thirty_day_BHAR_wo_filtering_null_Bloomberg.txt', sep=',', index_col=0, encoding='utf-8')

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


x_test, y_test_EP, y_test_bhar, y_test_ep90, y_test_eps = test_all['fileload'], test_all['env_t'], test_all['BHAR_t'], test_all['env_q'], test_all['EPS_q']
x_train, y_train_EP, y_train_bhar, y_train_ep90, y_train_eps = train_all['fileload'], train_all['env_t'], train_all['BHAR_t'], train_all['env_q'], train_all['EPS_q']
x_val, y_val_EP, y_val_bhar, y_val_ep90, y_val_eps = validation_all['fileload'], validation_all['env_t'], validation_all['BHAR_t'], validation_all['env_q'], validation_all['EPS_q']



#############################################################################
''' Load each document, clean files for stopwords, non-alphanumeric tokens,
and words shorter than 2 characters '''
#############################################################################

print("Loading text to memory!")

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

# for training set
text = cleanstr(x_train)
doc_training = np.array(text)

# for test set
text = cleanstr(x_test)
doc_test = np.array(text)

# for validation set
text = cleanstr(x_val)
doc_val = np.array(text)


#########################
''' Apply BOW to text'''
#########################

count = CountVectorizer()

## simple bow
x_train = count.fit_transform(doc_training)
x_test = count.transform(doc_test)
x_val = count.transform(doc_val)

x_train = np.array(x_train.toarray(), dtype='float32')
x_test = np.array(x_test.toarray(), dtype='float32')
x_val = np.array(x_val.toarray(), dtype='float32')


##################
''' Set y label'''
##################

## Note: If you want to run model for another label specify here

label = 'EPS_q'      ## << HERE 'env_t','env_q','EPS_q', 'BHAR_t'
y_train = y_train_eps    ## << HERE  y_train_EP, y_train_ep90, y_train_eps, y_train_bhar
y_test = y_test_eps      ## << HERE  y_test_EP, y_test_ep90, y_test_eps, y_test_bhar
y_val = y_val_eps        ## << HERE  y_val_EP, y_val_ep90, y_val_eps, y_val_bhar
print(label)

label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)



#############################################################################
''' Choose hyper-parameters and calculate baseline'''
#############################################################################

epoch = 50
batch = 32
input_dim = x_train.shape[1] #1000
num_workers = cpu_count()

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


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


accuracy, val_accuracy, manual_acc = [], [], []
loss, val_loss = [], []
fscores, f_m, f_m_val = [], [], []

precision_sc, precision_sc_val = [], []
recall_sc, recall_sc_val = [], []

f1_sc, f1_sc_val = [], []

tp, fp, fn = [], [], []
tp_val, fp_val, fn_val = [], [], []


###############################################
''' setup for financial labels (direct path)'''
###############################################

sequence_input = Input(shape=(input_dim,), dtype='float32')
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(sequence_input)
model = Model(sequence_input, preds)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[TruePositives(name='true_positives'),
                       TrueNegatives(name='true_negatives'),
                       FalseNegatives(name='false_negatives'),
                       FalsePositives(name='false_positives'),
                       ])

# checkpoint; models are stored after every epoch
filepath="weights-improvement-{epoch:02d}-{F1_score_val:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), monitor='F1_score_val', verbose=1, save_best_only=False, save_weights_only=True, mode='auto')#, save_freq='epoch')
#callbacks_list = [checkpoint]

history = model.fit(x_train, y_train,                
                    validation_data=(x_val, y_val),
                    epochs=epoch, batch_size=batch,
                    callbacks=[F1History(train=(x_train,y_train),validation=(x_val,y_val)),checkpoint]
                    )

loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])
loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()


## NOTE: TP;FP;FN seem to be wrong! Use ones calculated from reloading the model!'''
#tp.append(history.history['true_positives'])
#fp.append(history.history['false_positives'])
#fn.append(history.history['false_negatives'])

#tp_val.append(history.history['val_true_positives'])
#fp_val.append(history.history['val_false_positives'])
#fn_val.append(history.history['val_false_negatives'])

f1_sc.append(history.history['F1_score_train'])
f1_sc_val.append(history.history['F1_score_val'])

f1_sc = np.array(f1_sc).flatten()
f1_sc_val = np.array(f1_sc_val).flatten()


## saving loss and fscore for plots
diction = {'loss':loss,'val_loss':val_loss,'fscore':f1_sc,'val_fscore':f1_sc_val}
df = pd.DataFrame(diction, columns=['loss','val_loss','fscore','val_fscore'])
df.to_csv(savemodelpath+label+'loss_fscore.txt', sep=',', encoding='utf-8')


###############################
''' save best model fscores '''
###############################

best_validation_model = np.argmax(f1_sc_val)

best_trainings_fscore = f1_sc[np.argsort(f1_sc)[-1:]]
best_validation_fscore = f1_sc_val[np.argsort(f1_sc_val)[-1:]]
best_val_index = np.argmax(f1_sc_val)
best_train_index = np.argmax(f1_sc)

f1_sc = np.insert(f1_sc, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(f1_sc_val, 0, f_baseline_val, axis=0)


## We can extract the coefficients from the model (weights_preds)
weights_preds, bias_preds = model.get_weights()

pred_prob =  model.predict(x_train, verbose=1)  # gives probabilities
pred_class = [1 if x > 0.5 else 0 for x in pred_prob]

print(np.array(pred_prob).argmax())

doc_nb = np.array(pred_prob).argmax()
print("Document %s, Predicted Class: %s with prob. %s" % (doc_nb,pred_class[doc_nb], pred_prob[doc_nb]))
print("True class:", y_train[doc_nb])

first_doc = pd.DataFrame(x_train[doc_nb])
weigth = pd.DataFrame(weights_preds)
test = first_doc * weigth
words = pd.DataFrame(count.get_feature_names(), columns=['words'])
merged_test = test.merge(words, left_index=True, right_index=True)
merged_test = merged_test.sort_values(by=[0], ascending=False)
merged_test.rename(columns={0:'weights X document_tfidf'}, inplace=True)

### showing important learned words
### word weights have to be multiplied with
df_words_tfidf = pd.DataFrame(weights_preds.T, columns = count.get_feature_names())
tester = df_words_tfidf.T.sort_values(by=[0], ascending=False)
tester.rename(columns={0:'weights'}, inplace=True)
tester.to_csv(savemodelpath+label+'_words_weight.txt', sep=',', encoding='utf-8')


########################################
''' create plots for fscore and loss '''
########################################

#### PLOT USES F1_SC
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
plt.savefig(savemodelpath+label+'_BOW_f1.png')
plt.close()


plt.plot(range(1,epoch+1), loss, color='blue', marker='o', markersize=5)
plt.plot(range(1,epoch+1), val_loss, color='green', marker='s', markersize=5, linestyle ='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_loss.png')
plt.close()


############################
''' Re-load models '''
############################

## To calculate fscore history after each epoch,
## we reload each model. 

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
    ### fscore is calculated twice for soundness
    sequence_input = Input(shape=(input_dim,), dtype='float32')
    preds = Dense(1, activation='sigmoid',name='output')(sequence_input)
    model = Model(sequence_input, preds)
    model.load_weights(i)
    
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy',
              optimizer='adam', # opt
              metrics=[TruePositives(name='true_positives'),
                       TrueNegatives(name='true_negatives'),
                       FalseNegatives(name='false_negatives'),
                       FalsePositives(name='false_positives'),
                       ])

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


# run best model (highest validation score)'''
loadpath_best = model_loads[best_validation_model]

sequence_input = Input(shape=(input_dim,), dtype='float32')
preds = Dense(1, activation='sigmoid',name='output')(sequence_input)
model = Model(sequence_input, preds)
model.load_weights(loadpath_best)

# Compile model (required to make predictions)
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



############################
''' Get output '''
############################

print("")
print("Baseline train:", baseline_trains)
print("Baseline test:", baseline_test)
print("Baseline val:", baseline_vals)
print("")
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
print("")
print(">>      Copy in txt format!")

''' saving loss and fscore for plots '''
diction = {'pr_train':precision_train,'pr_val':precision_val,'rec_train':recall_train,'rec_val':recall_val}
df2 = pd.DataFrame(diction, columns=['pr_train','pr_val','rec_train','rec_val'])
df2.to_csv(savemodelpath+label+'recall_precision.txt', sep=',', encoding='utf-8')


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
plt.savefig(savemodelpath+label+'_BOW_precision.png')
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
plt.savefig(savemodelpath+label+'_BOW_recall.png')
plt.close()


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
plt.savefig(savemodelpath+label+'_BOW_f1.png')
plt.close()


plt.plot(range(1,epoch+1), loss, color='blue', marker='o', markersize=5)
plt.plot(range(1,epoch+1), val_loss, color='green', marker='s', markersize=5, linestyle ='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_loss.png')
plt.close()




#########################################################################################
''' model setup for environmental labels including Market Cap and Industry (path env) '''
#########################################################################################

label = 'env_q'      ## << HERE 'env_t','env_q','EPS_q'
y_train = y_train_ep90    ## << HERE  y_train_EP, y_train_ep90
y_test = y_test_ep90      ## << HERE  y_test_EP, y_test_ep90
y_val = y_val_ep90        ## << HERE  y_val_EP, y_val_ep90

label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)

epoch = 50
batch = 32
input_dim = x_train.shape[1] #1000
industry_dim = industry_train.shape[1]
num_workers = cpu_count()

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


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


accuracy, val_accuracy, manual_acc = [], [], []
loss, val_loss = [], []
fscores, f_m, f_m_val = [], [], []

precision_sc, precision_sc_val = [], []
recall_sc, recall_sc_val = [], []

f1_sc, f1_sc_val = [], []

tp, fp, fn = [], [], []
tp_val, fp_val, fn_val = [], [], []

##################################
''' setup for envionmental labels
    controlling for market cap and
    industry'''
##################################

sequence_input = Input(shape=(input_dim,), dtype='float32')
market_cap = Input(shape=(1,), dtype='float32')
industry = Input(shape=(industry_dim,), dtype='float32')

combined = Concatenate()([sequence_input, market_cap, industry])

preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,market_cap,industry], preds)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[TruePositives(name='true_positives'),
                       TrueNegatives(name='true_negatives'),
                       FalseNegatives(name='false_negatives'),
                       FalsePositives(name='false_positives'),
                       ])

# save checkpoint for reload 
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), verbose=1, save_best_only=False, save_weights_only=True, mode='auto')#, save_freq='epoch')
callbacks_list = [checkpoint]

history = model.fit([x_train,mktcap_train,industry_train], y_train,              
                    validation_data=([x_val,mktcap_val,industry_val], y_val), 
                    epochs=epoch, batch_size=batch,
                    callbacks=[checkpoint]
                    )

loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])


############################
''' Reloading the models '''
############################

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
    sequence_input = Input(shape=(input_dim,), dtype='float32')
    market_cap = Input(shape=(1,), dtype='float32')
    industry = Input(shape=(industry_dim,), dtype='float32')

    combined = Concatenate()([sequence_input, market_cap,industry])

    preds = Dense(1, activation='sigmoid',name='output',
                  #kernel_regularizer=regularizers.l2(1),
                  #bias_regularizer=regularizers.l2(1)
                  )(combined)

    model = Model([sequence_input,market_cap,industry
                   ], preds)

    model.load_weights(i)
    
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy',
              optimizer='adam', # opt
              metrics=[TruePositives(name='true_positives'),
                       TrueNegatives(name='true_negatives'),
                       FalseNegatives(name='false_negatives'),
                       FalsePositives(name='false_positives'),
                       #f1
                       ])

    # for training data
    y_pred =  model.evaluate([x_train,mktcap_train,industry_train
                              ], y_train, verbose=0)  # gives probabilities
    Fscore_train.append(y_pred)
    
    y_pred =  model.predict([x_train,mktcap_train,industry_train], verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)
    
    
    # for validation data
    y_prob =  model.evaluate([x_val,mktcap_val,industry_val
                              ], y_val, verbose=0)  # gives probabilities
    Fscore_val.append(y_prob)
    
    y_pred = model.predict([x_val,mktcap_val,industry_val], verbose=0)
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p, r = calc_metrics(val_preds, y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)
    

fscorepredict_train = np.array(fscorepredict_train).flatten()
fscorepredict_val_custom = np.array(fscorepredict_val_custom).flatten()


loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()

# saving loss and fscore for plots
diction = {'loss':loss,'val_loss':val_loss,'fscore':fscorepredict_train,'val_fscore':fscorepredict_val_custom}
df = pd.DataFrame(diction, columns=['loss','val_loss','fscore','val_fscore'])
df.to_csv(savemodelpath+label+'_loss_fscore.txt', sep=',', encoding='utf-8')


best_validation_model = np.argmax(fscorepredict_val_custom)

best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)

f1_sc = np.insert(fscorepredict_train, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(fscorepredict_val_custom, 0, f_baseline_val, axis=0)


## We extract the coefficients from the model (weights_preds)
weights_preds, bias_preds = model.get_weights()

pred_prob =  model.predict([x_train,mktcap_train,industry_train
                              ], verbose=1)  # gives probabilities
pred_class = [1 if x > 0.5 else 0 for x in pred_prob]

print(np.array(pred_prob).argmax())

doc_nb = np.array(pred_prob).argmax()
print("Document %s, Predicted Class: %s with prob. %s" % (doc_nb,pred_class[doc_nb], pred_prob[doc_nb]))
print("True class:", y_train[doc_nb])

first_doc = pd.DataFrame(x_train[doc_nb])
weigth = pd.DataFrame(weights_preds)
test = first_doc * weigth
words = pd.DataFrame(count.get_feature_names(), columns=['words'])
merged_test = test.merge(words, left_index=True, right_index=True)
merged_test = merged_test.sort_values(by=[0], ascending=False)
merged_test.rename(columns={0:'weights X document_tfidf'}, inplace=True)

### showing important learned words
### word weights have to be multiplied with
''' Note: trim weights to words only'''
new_length = len(weights_preds)-industry_dim-1
df_words_tfidf = pd.DataFrame(weights_preds.T[:,0:new_length], columns = count.get_feature_names())
tester = df_words_tfidf.T.sort_values(by=[0], ascending=False)
tester.rename(columns={0:'weights'}, inplace=True)
tester.to_csv(savemodelpath+label+'_words_weight.txt', sep=',', encoding='utf-8')


#### PLOT USES F1_SC since this is the correct Fscore 
#### (not the one calculated from batchwise nor tp,fp,fn)
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
''' here custom ticks ''' 
#plt.xticks(np.arange(0, epoch+1, 2.0))  #### way to show every second epoch on x-axis
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
plt.legend(['train','validation'], loc='lower left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_loss.png')
plt.close()


''' saving best train_preds and val_preds '''
loadpath_best = model_loads[best_train_index]

sequence_input = Input(shape=(input_dim,), dtype='float32')
market_cap = Input(shape=(1,), dtype='float32')
industry = Input(shape=(industry_dim,), dtype='float32')

combined = Concatenate()([sequence_input, market_cap,industry
                      ])
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,market_cap,industry
               ], preds)
model.load_weights(loadpath_best)

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy',
          optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
                   #f1
                   ])

y_pred_train = model.predict([x_train,mktcap_train,industry_train])
train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
f_train,pr_train, re_train = calc_metrics(train_preds, y_train)

saved_train_preds = pd.DataFrame(train_preds)
saved_train_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_train_preds.txt',sep=',',encoding='utf-8')


####################################################
''' load best models and calculate metrics '''
###################################################

loadpath_best = model_loads[best_val_index]

sequence_input = Input(shape=(input_dim,), dtype='float32')
market_cap = Input(shape=(1,), dtype='float32')
industry = Input(shape=(industry_dim,), dtype='float32')

combined = Concatenate()([sequence_input, market_cap,industry
                      ])
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,market_cap,industry
               ], preds)
model.load_weights(loadpath_best)

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy',
          optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
                   #f1
                   ])

y_pred_val = model.predict([x_val,mktcap_val,industry_val])
val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
f_val,pr_val, re_val = calc_metrics(val_preds, y_val)

saved_val_preds = pd.DataFrame(val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_val_preds.txt',sep=',',encoding='utf-8')

bootstrap_val = TP_FP_FN_forBootstrap(val_preds,y_val)
bootstrap_val.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_val.txt',sep=',',encoding='utf-8')

# run best model (highest validation score)
loadpath_best = model_loads[best_validation_model]

sequence_input = Input(shape=(input_dim,), dtype='float32')
market_cap = Input(shape=(1,), dtype='float32')
industry = Input(shape=(industry_dim,), dtype='float32')

combined = Concatenate()([sequence_input, market_cap,industry
                      ])
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,market_cap,industry
               ], preds)
model.load_weights(loadpath_best)

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy',
          optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
                   ])

y_pred_test = model.predict([x_test,mktcap_test,industry_test])

test_preds = [1 if x > 0.5 else 0 for x in y_pred_test]
f_test,pr_test, re_test = calc_metrics(test_preds, y_test)

bootstrap_test = TP_FP_FN_forBootstrap(y_pred_test,y_test)
bootstrap_test.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_test.txt',sep=',',encoding='utf-8')


print("")
print("Baseline train:", baseline_trains)
print("Baseline test:", baseline_test)
print("Baseline val:", baseline_vals)
print("")
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
print(">>      Copy in txt format!")





##################################################################################
''' Environmental Performance + Text ->  Financial Performance (Combined path) '''
##################################################################################

label = 'label_eps'      ## << HERE 'label_eps', 'BHAR_t'
y_train = y_train_eps    ## << HERE  y_train_eps, y_train_bhar
y_test = y_test_eps      ## << HERE  y_test_eps, y_test_bhar
y_val = y_val_eps        ## << HERE y_val_eps, y_val_bhar

golden = 'env_q'                  ## << HERE 'env_t','env_q'
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


accuracy, val_accuracy, manual_acc = [], [], []
loss, val_loss = [], []
fscores, f_m, f_m_val = [], [], []

precision_sc, precision_sc_val = [], []
recall_sc, recall_sc_val = [], []

f1_sc, f1_sc_val = [], []

tp, fp, fn = [], [], []
tp_val, fp_val, fn_val = [], [], []


#######################
''' model setup '''
#######################

sequence_input = Input(shape=(input_dim,), dtype='float32')
env_perf = Input(shape=(1,), dtype='float32')
combined = Concatenate()([sequence_input, env_perf])
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,env_perf], preds)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[TruePositives(name='true_positives'),
                       TrueNegatives(name='true_negatives'),
                       FalseNegatives(name='false_negatives'),
                       FalsePositives(name='false_positives'),
                       ])

# save checkpoints
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
    sequence_input = Input(shape=(input_dim,), dtype='float32')
    env_perf = Input(shape=(1,), dtype='float32')

    combined = Concatenate()([sequence_input, env_perf])

    preds = Dense(1, activation='sigmoid',name='output',
                  #kernel_regularizer=regularizers.l2(1),
                  #bias_regularizer=regularizers.l2(1)
                  )(combined)

    model = Model([sequence_input,env_perf
                   ], preds)

    model.load_weights(i)
    
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[TruePositives(name='true_positives'),
                       TrueNegatives(name='true_negatives'),
                       FalseNegatives(name='false_negatives'),
                       FalsePositives(name='false_positives'),
                       ])

    # for training data
    y_pred =  model.evaluate([x_train,golden_env_train
                              ], y_train, verbose=0)  # gives probabilities
    Fscore_train.append(y_pred)
    
    y_pred =  model.predict([x_train,golden_env_train], verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p_tr, r_tr = calc_metrics(train_preds, y_train)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    cm = f1_score(y_train, train_preds)
    fscorepredict_train.append(cm)
    
    
    # for validation data
    y_prob =  model.evaluate([x_val,golden_env_val
                              ], y_val, verbose=0)  # gives probabilities
    Fscore_val.append(y_prob)
    
    y_pred = model.predict([x_val,golden_env_val], verbose=0)
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    cm, p, r = calc_metrics(val_preds, y_val)
    precision_val.append(p)
    recall_val.append(r)    
    
    fscorepredict_val_custom.append(cm)
    cm = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(cm)
    

fscorepredict_train = np.array(fscorepredict_train).flatten()
fscorepredict_val_custom = np.array(fscorepredict_val_custom).flatten()


best_validation_model = np.argmax(fscorepredict_val_custom)

best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)

f1_sc = np.insert(fscorepredict_train, 0, f_baseline_train, axis=0)
f1_sc_val = np.insert(fscorepredict_val_custom, 0, f_baseline_val, axis=0)


## We extract the coefficients from the model (weights_preds)
weights_preds, bias_preds = model.get_weights()


pred_prob =  model.predict([x_train,golden_env_train
                              ], verbose=1)  # gives probabilities
pred_class = [1 if x > 0.5 else 0 for x in pred_prob]

print(np.array(pred_prob).argmax())

doc_nb = np.array(pred_prob).argmax()
print("Document %s, Predicted Class: %s with prob. %s" % (doc_nb,pred_class[doc_nb], pred_prob[doc_nb]))
print("True class:", y_train[doc_nb])

first_doc = pd.DataFrame(x_train[doc_nb])
weigth = pd.DataFrame(weights_preds)
test = first_doc * weigth
words = pd.DataFrame(count.get_feature_names(), columns=['words'])
merged_test = test.merge(words, left_index=True, right_index=True)
merged_test = merged_test.sort_values(by=[0], ascending=False)
merged_test.rename(columns={0:'weights X document_tfidf'}, inplace=True)

### showing important learned words
### word weights have to be multiplied with
''' Note: trim weights to words only'''
new_length = len(weights_preds)-1
df_words_tfidf = pd.DataFrame(weights_preds.T[:,0:new_length], columns = count.get_feature_names())
tester = df_words_tfidf.T.sort_values(by=[0], ascending=False)
tester.rename(columns={0:'weights'}, inplace=True)
tester.to_csv(savemodelpath+label+'_words_weight.txt', sep=',', encoding='utf-8')


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
''' here custom ticks ''' 
#plt.xticks(np.arange(0, epoch+1, 2.0))  #### way to show every second epoch on x-axis
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower right')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_f1.png')
plt.close()


loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()

plt.plot(range(1,epoch+1), loss, color='blue', marker='o', markersize=5)
plt.plot(range(1,epoch+1), val_loss, color='green', marker='s', markersize=5, linestyle ='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.xticks(range(0, epoch+1))
plt.xlim(-1, epoch+2)
plt.legend(['train','validation'], loc='lower left')
#plt.show()
plt.savefig(savemodelpath+label+'_BOW_loss.png')
plt.close()


# saving best train_preds and val_preds
loadpath_best = model_loads[best_train_index]

sequence_input = Input(shape=(input_dim,), dtype='float32')
env_perf = Input(shape=(1,), dtype='float32')

combined = Concatenate()([sequence_input, env_perf
                      ])
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,env_perf
               ], preds)
model.load_weights(loadpath_best)

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy',
          optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
                   #f1
                   ])

y_pred_train = model.predict([x_train,golden_env_train])
train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
f_train,pr_train, re_train = calc_metrics(train_preds, y_train)

saved_train_preds = pd.DataFrame(train_preds)
saved_train_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_train_preds.txt',sep=',',encoding='utf-8')

#########################
''' reload best model '''
#########################

loadpath_best = model_loads[best_val_index]

sequence_input = Input(shape=(input_dim,), dtype='float32')
env_perf = Input(shape=(1,), dtype='float32')

combined = Concatenate()([sequence_input, env_perf
                      ])
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,env_perf
               ], preds)
model.load_weights(loadpath_best)

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy',
          optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
                   #f1
                   ])

y_pred_val = model.predict([x_val,golden_env_val])
val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
f_val,pr_val, re_val = calc_metrics(val_preds, y_val)

saved_val_preds = pd.DataFrame(val_preds)
saved_val_preds.to_csv(savemodelpath+label+'_mkt_ind_saved_val_preds.txt',sep=',',encoding='utf-8')

bootstrap_val = TP_FP_FN_forBootstrap(val_preds,y_val)
bootstrap_val.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_val.txt',sep=',',encoding='utf-8')

## running best model (highest validation score)
loadpath_best = model_loads[best_validation_model]
sequence_input = Input(shape=(input_dim,), dtype='float32')
env_perf = Input(shape=(1,), dtype='float32')

combined = Concatenate()([sequence_input, env_perf
                      ])
preds = Dense(1, activation='sigmoid',name='output',
              #kernel_regularizer=regularizers.l2(1),
              #bias_regularizer=regularizers.l2(1)
              )(combined)

model = Model([sequence_input,env_perf
               ], preds)
model.load_weights(loadpath_best)

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy',
          optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
                   ])

y_pred_test = model.predict([x_test,golden_env_test])

test_preds = [1 if x > 0.5 else 0 for x in y_pred_test]
f_test,pr_test, re_test = calc_metrics(test_preds, y_test)

bootstrap_test = TP_FP_FN_forBootstrap(y_pred_test,y_test)
bootstrap_test.to_csv(savemodelpath+label+'_mkt_ind_bootstrap_test.txt',sep=',',encoding='utf-8')


print("")
print("Baseline train:", baseline_trains)
print("Baseline test:", baseline_test)
print("Baseline val:", baseline_vals)
print("")
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
print(">>      Copy in txt format!")

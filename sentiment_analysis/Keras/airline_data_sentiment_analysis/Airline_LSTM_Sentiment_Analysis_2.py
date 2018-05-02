
# coding: utf-8

# In[4]:


from __future__ import print_function
from keras.utils import plot_model
from keras.activations import elu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import keras
from pandas import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Bidirectional,Embedding, MaxPooling1D
from keras.layers import Conv1D, AveragePooling1D, Concatenate, concatenate
from keras.layers import LSTM, Flatten, TimeDistributed, SpatialDropout1D, GlobalMaxPooling1D, Masking
from keras.optimizers import rmsprop, adam
from sklearn.metrics import accuracy_score, regression
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint, CSVLogger
import re
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import keras.backend as K
import multiprocessing
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from keras.layers.core import Dense, Dropout, Flatten
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer


# In[37]:


from keras.layers import concatenate as concatLayer


# In[5]:


df = pd.read_csv('/mydata/clean_tweet.csv',index_col=0)
df.head()


# In[6]:


df['airline'] = pd.factorize(df['airline'].values)[0]


# In[7]:


df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


# In[8]:


max_features = 10000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X)


# In[9]:


X.shape


# In[13]:


#Spliting Train, Test, Valid by 60,20,20
Y = df['airline_sentiment']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train,
                                                    test_size = 0.15, random_state = 42)


# In[14]:


train_ind = Y_train.index.values.tolist()
test_ind = Y_test.index.values.tolist()
valid_ind = Y_valid.index.values.tolist()


# In[15]:


#Saving "Airline" column splitting for inputting as a seperate Dense layer
X_train_ = df['airline'][Y_train.index]
X_test_ = df['airline'][Y_test.index]
X_valid_ = df['airline'][Y_valid.index]


# In[16]:


#Converting the output to categorical classes
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_valid = to_categorical(Y_valid)


# In[17]:


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()


# In[18]:


df['text'] = df['text'].apply(clean_str)
corpus = df['text'].tolist()
corpus[:10]


# In[19]:

x= len(corpus)
print(str('Corpus size '))
print(str(x))


# In[20]:


# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
    tokenized_corpus.append(tokens)


# In[21]:


# Gensim Word2Vec model
vector_size = 250
window_size = 5

# Create Word2Vec
word2vec = Word2Vec(sentences=tokenized_corpus,
                    #max_vocab_size=max_features,
                    #sample=1e-5,
                    size=vector_size, 
                    window=window_size,
                    #hs=1,
                    negative=20,
                    iter=100,
                    seed=1,
                    workers=multiprocessing.cpu_count())


# In[22]:


# Copy word vectors and delete Word2Vec model  and original corpus to save memory
X_vecs = word2vec.wv
#del word2vec
#del corpus


# In[23]:


# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))
    
print('Average tweet length: {}'.format(avg_length / float(len(tokenized_corpus))))
print('Max tweet length: {}'.format(max_length))


# In[24]:


X_vecs.vocab['cup'].index


# In[25]:


X_vecs.vocab['tea'].index


# In[26]:


max_tweet_length = 31


# In[27]:


def word2idx(word):
    return X_vecs.vocab[word].index
def idx2word(idx):
    return X_vecs.index2word[idx]


# In[28]:


def Word2vec_corpus(train_ind):
    train_x = list( tokenized_corpus[i] for i in train_ind )
    train_X = np.zeros([len(train_x), max_tweet_length], dtype=np.int32)
    for i, sentence in enumerate(train_x):
        for t, word in enumerate(sentence[:-1]):
            if word in X_vecs.vocab:
                train_X[i, t] = word2idx(word)
            else:
                pass
    return train_X


# In[29]:


X_train = Word2vec_corpus(train_ind)
X_test = Word2vec_corpus(test_ind)
X_valid = Word2vec_corpus(valid_ind)


# In[30]:


X_train.shape


# In[31]:


# Model3
embed_dim = 512
lstm_out = 16

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(32, dropout=0.5,
                             return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Dropout(0.3))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',
              optimizer=rmsprop(lr=1e-4),metrics = ['accuracy'])
print(model.summary())
#checkpoint = ModelCheckpoint(filepath='/output/bestModel_stacked_w2v_3.hdfs', save_weights_only=False,
#                             monitor='val_loss',save_best_only=True)
batch_size = 128
#plot_model(model, to_file='bestModel_stacked_w2v_3.png')


# ### Uncomment the next cell for training

# In[26]:


#model.fit(X_train, Y_train, epochs = 80,validation_data=(X_valid,Y_valid),
#          callbacks=[checkpoint], batch_size=batch_size, verbose = 2)


# In[26]:


#model.load_weights('/output/bestModel_stacked_w2v_3.hdfs')
pred = model.predict(X_test,batch_size=batch_size)
acc_test = accuracy_score(Y_test.argmax(axis=1),pred.argmax(axis=1))
print('Model1 accuracy' + str(acc_test))


# #Testing conv1d as LSTM is overfitting at 80 test accuracy

# In[34]:


# Model4

#CNN_no_dropout\n"
print('Build CNN_no_dropout model...')
embed_dim = 512
input1 = Input(shape=(max_tweet_length,))
Emb = Embedding(max_features, embed_dim)(input1)
#model.add(SpatialDropout1D(0.4))
m = Conv1D(32,(2))(Emb)
#model.add(Conv1D(16,(2)))
m = AveragePooling1D(16,2)(m)
m = Dropout(0.5)(m)
m = Flatten()(m)
m = Dense(64)(m)
m = Dropout(0.5)(m)
output = Dense(3,activation='softmax')(m)
m = Model(inputs=[input1],outputs=[output])
m.compile(loss = 'categorical_crossentropy',
          optimizer=rmsprop(lr=1e-4),metrics = ['accuracy'])
rlStop = ReduceLROnPlateau(monitor='val_loss',min_lr=1e-6,patience=4,factor=0.5)
#checkpoint = ModelCheckpoint(filepath='/output/CNN_stacked_w2v.hdfs', save_weights_only=False,
#                             monitor='val_loss',save_best_only=True)
batch_size = 64
#plot_model(m, to_file='CNN_stacked_w2v.png')


# ### Uncomment the next cell for training

# In[ ]:


#m.fit(X_train, Y_train, epochs = 350,validation_data=(X_valid,Y_valid),callbacks=[checkpoint,rlStop],
#          batch_size=batch_size, verbose = 2)


# In[28]:


#m.load_weights('/output/CNN_stacked_w2v.hdfs')
pred = m.predict(X_test,batch_size=batch_size)
acc_test = accuracy_score(Y_test.argmax(axis=1),pred.argmax(axis=1))
print('Model2 accuracy' + str(acc_test))


# In[38]:


#Model7
batch_size = 64
n_bilstm_1 = 8
n_bilstm_2 = 16
drop_bilstm = 0.25
embed_dim = 512
input_layer = Input(shape=(max_tweet_length,))
embedding_layer = Embedding(max_features, embed_dim)(input_layer)

bi_lstm_1 = Bidirectional(LSTM(n_bilstm_1, dropout=drop_bilstm, return_sequences=True,
                               name='bi_lstm_1'))(embedding_layer)
bi_lstm_2 = Bidirectional(LSTM(n_bilstm_2, dropout=drop_bilstm, return_sequences=True,
                               name='bi_lstm_2'))(embedding_layer)
concat = concatLayer([bi_lstm_1, bi_lstm_2])
densor = Dense(64)(concat)
flat = Flatten()(densor)
output = Dense(3, activation='softmax', name='output')(flat)
model = Model(input_layer, output)
model.compile(loss = 'categorical_crossentropy',
              optimizer=rmsprop(lr=1e-4),metrics = ['accuracy'])
rlStop = ReduceLROnPlateau(monitor='val_loss',min_lr=1e-6,patience=4,factor=0.5)
#checkpoint = ModelCheckpoint(filepath='/output/Branched_BiLSTM.hdfs', save_weights_only=False,
#                            monitor='val_loss',save_best_only=True)
batch_size = 64
#plot_model(model, to_file='Branched_BiLSTM.png')


# ### Uncomment the next cell for training

# In[103]:

print('Begin modeling Branched_BiLSTM Model accuracy')
model.fit(X_train, Y_train, epochs = 20,validation_data=(X_valid,Y_valid),
          batch_size=batch_size, verbose = 1)


# In[105]:


#model.load_weights('/output/Branched_BiLSTM.hdfs')
pred = model.predict(X_test,batch_size=batch_size)
acc_test = accuracy_score(Y_test.argmax(axis=1),pred.argmax(axis=1))
print('Branched_BiLSTM Model accuracy' + str(acc_test))

# ### Testing Character level modeling combined with the tokenized one

# In[68]:


chars = sorted(list(set(' '.join(corpus))))
mapping = dict((c, i) for i, c in enumerate(chars))


# In[69]:


def char_encode(str):
    encoded_seq = [mapping[char] for char in str]
    return encoded_seq


# In[70]:


df['encoded_text'] = df['text'].apply(char_encode)


# In[71]:


df.head()


# In[72]:


max_char_len = max([len(i) for i in df['encoded_text']])
max_char_len


# In[73]:


chars_len = len(chars)
chars_len


# In[74]:


def zero_pad(list1):
    x = np.pad(list1, (0,max_char_len-len(list1)), 'constant', constant_values=0)
    return x


# In[75]:


df['encoded_text'] = df['encoded_text'].apply(zero_pad)


# In[76]:


# loading the  Character vector input for train, test, valid
V_train = np.array([df['encoded_text'][x] for x in train_ind])
V_test  = np.array([df['encoded_text'][x] for x in test_ind])
V_valid = np.array([df['encoded_text'][x] for x in valid_ind])


# In[100]:


#Model8
n_bilstm_1 = 16
n_bilstm_2 = 16
drop_bilstm = 0.25
embed_dim = 512

input_1 = Input(shape=(max_tweet_length,),name = 'Word_embedding_Input')
embedding_1 = Embedding(max_features, embed_dim)(input_1)
bi_lstm_1 = Bidirectional(LSTM(n_bilstm_1, dropout=drop_bilstm, return_sequences=True,
                               name='bi_lstm_1'))(embedding_1)

input_2 = Input(shape=(max_char_len,),name = 'Character_embedding_Input')
embedding_2 = Embedding(chars_len, 50)(input_2)
bi_lstm_2 = Bidirectional(LSTM(n_bilstm_2, dropout=drop_bilstm, return_sequences=True,
                               name='bi_lstm_2'))(embedding_2)
concat = concatLayer([bi_lstm_1, bi_lstm_2],axis=1)
densor = Dense(64)(concat)
densor = Dropout(0.5)(densor)
flat = Flatten()(densor)
output = Dense(3, activation='softmax', name='output')(flat)
model = Model([input_1, input_2], output)
model.compile(loss = 'categorical_crossentropy',
              optimizer=rmsprop(lr=1e-4),metrics = ['accuracy'])
rlStop = ReduceLROnPlateau(monitor='val_loss',min_lr=1e-6,patience=4,factor=0.5)
#checkpoint = ModelCheckpoint(filepath='/output/Branched_BiLSTM_2_Input.hdfs', save_weights_only=False,
#                             monitor='val_loss',save_best_only=True)
batch_size = 64
#plot_model(model, to_file='Branched_BiLSTM_2_Input.png')


# ### Uncomment the next cell for training

# In[99]:

print('Begin modeling Branched_BiLSTM_2_Input Model accuracy')
model.fit([X_train, V_train],Y_train,epochs = 20,validation_data=([X_valid,V_valid],Y_valid),
          batch_size=batch_size, verbose = 1)


# In[102]:


#model.load_weights('/output/Branched_BiLSTM_2_Input.hdfs')
pred = model.predict([X_test,V_test],batch_size=batch_size)
acc_test = accuracy_score(Y_test.argmax(axis=1),pred.argmax(axis=1))
print('Branched_BiLSTM_2_Input Model accuracy' + str(acc_test))


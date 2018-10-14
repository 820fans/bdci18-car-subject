
# coding: utf-8

# In[1]:


import pandas as pd

train = pd.read_csv("train.csv")
target = pd.read_csv("test_public.csv")


# In[2]:


# encode label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['subject'])
train.loc[:,'subject_no'] = pd.Series(le.transform(train['subject']), index=train.index)
train.head()


# In[3]:


# 简短的weight
from sklearn.utils import class_weight
import numpy as np
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train['subject_no']),
                                                 train['subject_no'])
class_weights


# In[4]:


from sklearn.preprocessing import LabelBinarizer
enc = LabelBinarizer()
enc.fit([0,1,2,3,4,5,6,7,8,9])

subject_onehot =enc.transform(train['subject_no'])


# In[5]:


import jieba
content_col = train['content'].apply(lambda x: [word for word in jieba.cut(x)])
content_col_tar = target['content'].apply(lambda x: [word for word in jieba.cut(x)])
train.loc[:, 'content_words'] = pd.Series([" ".join(words) for words in content_col], index=train.index)
target.loc[:, 'content_words'] = pd.Series([" ".join(words) for words in content_col_tar], index=target.index)
# content shape
print(len(content_col), max([len(x) for x in content_col]))
print(len(content_col_tar), max([len(x) for x in content_col_tar]))
max_length=max([len(x) for x in content_col])


# In[6]:


vocab = set()
for words in content_col.values:
    for word in words:
        vocab.add(word)
for words in content_col_tar.values:
    for word in words:
        vocab.add(word)
vocab_size = len(vocab)


# In[7]:


from keras.preprocessing.text import one_hot
encoded_train = [one_hot(d, vocab_size) for d in train['content_words'].values]
encoded_train
encoded_target = [one_hot(d, vocab_size) for d in target['content_words'].values]


# In[8]:


from keras.preprocessing.sequence import pad_sequences
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_target = pad_sequences(encoded_target, maxlen=max_length, padding='post')


# In[9]:


from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Embedding, Input
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.layers import Input, Dense, Masking, LSTM, Bidirectional

OUTPUT_SIZE = 10
CELL_SIZE = 50
BATCH_SIZE = 24
LR = 0.01
epoche = 20 # 训练100轮

# define the model 定义模型
from attention_lstm import model_attention_applied_after_lstm

model = model_attention_applied_after_lstm()

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length, mask_zero=True))
model.add(Bidirectional(LSTM(CELL_SIZE, dropout=0.1, recurrent_dropout=0.1)))
model.add(Dense(OUTPUT_SIZE, activation='softmax'))
# optimizer
model.compile(optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'])

# summarize the model 打印模型信息
print(model.summary())


# In[10]:


from keras.callbacks import ModelCheckpoint
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

X_train, y_train = padded_train, subject_onehot
# Fit the model
model.fit(X_train, y_train,
  epochs=epoche,
  batch_size=BATCH_SIZE,
  validation_split=0.2, shuffle=True, 
  verbose=1, callbacks=callbacks_list,
          class_weight= class_weights
  )


# In[12]:


doc_predicted = model.predict(padded_target)
prdc= enc.inverse_transform(doc_predicted).reshape(len(content_col_tar))


# In[13]:


submit = pd.DataFrame(columns=['content_id'])
submit['content_id']=target['content_id']
submit['subject'] = pd.Series(le.inverse_transform(prdc), index=target.index)
submit['sentiment_value'] = 0
submit['sentiment_word'] = ""
submit.to_csv("submit_emb1008.csv", index=False)


# In[11]:





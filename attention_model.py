
# coding: utf-8

# In[1]:


import pandas as pd

train = pd.read_csv("train.csv")
target = pd.read_csv("test_public.csv")
train['sentiment_value'].unique()


# In[2]:


# 检查情感极性数据比例情况
train['sentiment_value'].value_counts()
# 情感极性存在不平衡的情况 


# In[3]:


# 主题
train['subject'].value_counts()


# In[4]:


# 主题也存在不平衡的情况

# encode label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['subject'])
le.transform(["动力", "配置", "空间"])


# In[5]:


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
le.inverse_transform([0, 3, 7])
# 成功label


# In[6]:


train.loc[:,'subject_no'] = pd.Series(le.transform(train['subject']), index=train.index)
train.head()


# In[7]:
train['content'] = train['content'] + train['subject']
target['content'] = target['content']

import jieba
content_col = train['content'].apply(lambda x: [word for word in jieba.cut(x)])
content_col_tar = target['content'].apply(lambda x: [word for word in jieba.cut(x)])
train.loc[:, 'content_words'] = pd.Series(content_col, index=train.index)
target.loc[:, 'content_words'] = pd.Series(content_col_tar, index=target.index)
# content shape
#print(len(content_col), max([len(x) for x in content_col]))
#print(len(content_col_tar), max([len(x) for x in content_col_tar]))
max_length=max([len(x) for x in content_col])


# In[8]:


from gensim.models import word2vec
f_model = "/public/liangy/Data/word2vec/weibo_3g_model/weibo_freshdata_m.2016-10-07"
wmodel = word2vec.Word2Vec.load(f_model)


# In[9]:


for words in content_col.values:
    print(wmodel.wv[words])
    print(len(wmodel.wv.vocab))
    break


# In[23]:


# print('因为' in model.wv.vocab)
import numpy as np


dim1 = len(content_col)
dim2 = max_length
dim3 = 200
dtrain = np.zeros([dim1, dim2, dim3], dtype=np.float32)

cnt = 0
for words in content_col.values:
    for j, word in enumerate(words):
        try:
            dtrain[cnt][j] = wmodel.wv[word]
        except:
            pass
    cnt += 1
dtrain.shape


# In[25]:


dim1 = len(content_col_tar)
dim2 = max_length
dim3 = 200
dtarget = np.zeros([dim1, dim2, dim3], dtype=np.float32)

cnt = 0
for words in content_col_tar.values:
    for j, word in enumerate(words):
        try:
            dtarget[cnt][j] = wmodel.wv[word]
        except:
            pass
    cnt += 1
dtarget.shape


# In[10]:


train['subject_no'].values


# In[26]:


import numpy as np
from sklearn.preprocessing import LabelBinarizer

enc = LabelBinarizer()
enc.fit([0,1,2,3,4,5,6,7,8,9])

subject_onehot =enc.transform(train['subject_no'])
prdc= enc.inverse_transform(np.matrix([
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0]
])).reshape(2)
le.inverse_transform(prdc)


y_classes = train['subject']
# Create integer based labels Series
y_integers = le.transform(list(y_classes))

# Create dict of labels : integer representation
labels_and_integers = dict(zip(y_classes, y_integers))

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
sample_weights = compute_sample_weight('balanced', y_integers)

class_weights = dict(zip(le.transform(list(le.classes_)), class_weights))

submit = pd.DataFrame(columns=['content_id'])
submit['content_id']=target['content_id']


# In[40]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(padded_train, subject_onehot, test_size=0.9, random_state=42)#
X_train, y_train = dtrain, subject_onehot

from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.layers import Input, Dense, Masking, LSTM, Bidirectional

TIME_STEPS = max_length  # same as the height of the image
INPUT_SIZE = 200  # same as the width of the image
OUTPUT_SIZE = 10
CELL_SIZE = 100
BATCH_SIZE = 16
LR = 0.01
epoche = 50 # 训练50轮

# build RNN model
from attention_lstm import model_attention_applied_after_lstm

model = model_attention_applied_after_lstm(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE)

from keras import metrics
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# optimizer
model.compile(optimizer='adam',
      loss='categorical_crossentropy',
      metrics=[f1])

# plot_model(model, show_shapes=True, to_file='model_lstm.png')
model.fit(X_train, y_train,
  epochs=epoche,
  batch_size=BATCH_SIZE,
  validation_split=0.2, shuffle=True,
  verbose=1, class_weight=class_weights
  )

model.save("model/bilstm.model")

#loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
doc_predicted = model.predict(dtarget)
prdc= enc.inverse_transform(doc_predicted).reshape(len(content_col_tar))

submit['subject'] = pd.Series(le.inverse_transform(prdc), index=target.index)
submit['sentiment_value'] = 0
submit['sentiment_word'] = ""
submit.to_csv("submit_atte-with-label.csv", index=False)


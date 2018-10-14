
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
senti_le = LabelEncoder()
senti_le.fit(train['sentiment_value'])
train.loc[:,'sentiment_no'] = pd.Series(senti_le.transform(train['sentiment_value']), index=train.index)
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


import jieba
def processing_data(data):
    word = jieba.cut(data)
    return ' '.join(word)


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

all_data = pd.concat([train, target])
all_data['content_words'] = all_data['content'].map(processing_data)

print('TfidfVectorizer')
tf = TfidfVectorizer(ngram_range=(1,2),min_df=1)
discuss_tf = tf.fit_transform(all_data['content_words'])

print('HashingVectorizer')
ha = HashingVectorizer(ngram_range=(1,1),lowercase=False)
discuss_ha = ha.fit_transform(all_data['content_words'])

vec_data = hstack((discuss_tf,discuss_ha)).tocsr()

# vec_data = discuss_tf
train_size = train.shape[0]
# 分割
train_weight, target_weight = vec_data[:train_size], vec_data[train_size:]


# In[6]:


train_y = train['subject_no'].values
senti_y = train['sentiment_no'].astype(int).values
train['sentiment_value'].values


# In[7]:


senti_y


# In[8]:


import sklearn.metrics as metrics
def micro_avg_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')

N=10
# y_train_al = np.zeros_like(train_y, dtype='float64')
y_target = np.zeros((target.shape[0], N))
y_senti  = np.zeros((target.shape[0], N))


# In[15]:


import xgboost as xgb
dtarget = xgb.DMatrix(target_weight)

senti_param = {'max_depth': 8, 'eta':0.05, 'eval_metric':'merror', 
             'max_delta_step': 0, 'subsample': 1, 'alpha': 1, 'lambda': 1, 'scale_pos_weight': 1,
    'n_estimators': 728, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 
         'silent':1, 'objective':'multi:softmax', 'num_class':3, 'seed': 1254}  # 参数
param = {'max_depth': 9, 'eta':0.05, 'eval_metric':'merror',
             'max_delta_step': 0, 'subsample': 1, 'alpha': 1, 'lambda': 1, 'scale_pos_weight': 1,
    'n_estimators': 728, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 
         'silent':1, 'objective':'multi:softmax', 'num_class':10, 'seed': 1993}  # 参数


# In[16]:


from sklearn.model_selection import StratifiedKFold

N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=1684).split(train_weight, train_y)


# In[17]:


acc = 0
vcc = 0
for i ,(train_fold,test_fold) in enumerate(skf):
    
    X_train, X_validate, y_train, y_validate, senti_train, senti_validate = train_weight[train_fold, :], train_weight[test_fold, :], train_y[train_fold], train_y[test_fold], senti_y[train_fold], senti_y[test_fold]
    
    # 针对情感的分类
    senti_dtrain = xgb.DMatrix(X_train, label=senti_train)
    senti_dvalidate = xgb.DMatrix(X_validate, label=senti_validate)
    evallist  = [(senti_dtrain,'senti_train'), (senti_dvalidate,'senti_validate')]
    num_round = 500  # 循环次数
    clf = xgb.train(senti_param, senti_dtrain, num_round, evallist, early_stopping_rounds=20)
    val_ = clf.predict(senti_dvalidate)
    acc += micro_avg_f1(senti_validate, val_)
    result = clf.predict(dtarget)
    y_senti[:, i] = result
    
    # 针对主题的分类
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalidate = xgb.DMatrix(X_validate, label=y_validate)
    evallist  = [(dtrain,'train'), (dvalidate,'validate')] 
    num_round = 500 # 循环次数
    clf = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20)
    
    val_1 = clf.predict(dvalidate)
    vcc += micro_avg_f1(y_validate, val_1)
    result = clf.predict(dtarget)
    y_target[:, i] = result

print(acc/N)
print(vcc/N)


# In[18]:


from collections import Counter
res_senti = []
for i in range(y_senti.shape[0]):
    tmp = []
    for j in range(N):
        tmp.append(int(y_senti[i][j]))
    word_counts = Counter(tmp)
    yes = word_counts.most_common(1)
    res_senti.append(senti_le.inverse_transform(yes[0][0]))
res_senti


# In[19]:


from collections import Counter
res_2 = []
for i in range(y_target.shape[0]):
    tmp = []
    for j in range(N):
        tmp.append(int(y_target[i][j]))
    word_counts = Counter(tmp)
    yes = word_counts.most_common(1)
    res_2.append(le.inverse_transform(yes[0][0]))
res_2


# In[20]:


submit = pd.DataFrame(columns=['content_id'])
submit['content_id']=target['content_id']
submit['subject'] = pd.Series(res_2, index=target.index)
submit['sentiment_value'] = pd.Series(res_senti, index=target.index)
submit['sentiment_word'] = ""
submit.to_csv("submit_kf_xgb.csv", index=False)


# In[ ]:





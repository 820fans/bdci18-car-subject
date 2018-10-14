
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


# In[ ]:





# In[4]:


import jieba
content_col = train['content'].apply(lambda x: [word for word in jieba.cut(x)])
content_col_tar = target['content'].apply(lambda x: [word for word in jieba.cut(x)])
train.loc[:, 'content_words'] = pd.Series([" ".join(words) for words in content_col], index=train.index)
target.loc[:, 'content_words'] = pd.Series([" ".join(words) for words in content_col_tar], index=target.index)
# content shape
print(len(content_col), max([len(x) for x in content_col]))
print(len(content_col_tar), max([len(x) for x in content_col_tar]))
max_length=max([len(x) for x in content_col])


# In[5]:


vocab = set()
for words in content_col.values:
    for word in words:
        vocab.add(word)
for words in content_col_tar.values:
    for word in words:
        vocab.add(word)
vocab_size = len(vocab)


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train["content_words"]))
weight = tfidf.toarray()
print(weight)
test_tfidf = tfidftransformer.transform(vectorizer.transform(target["content_words"]))
test_weight = test_tfidf.toarray()
print(test_weight.shape)


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(weight, train['subject_no'].values, test_size=0.01, random_state=42)#


# In[32]:


import xgboost as xgb


from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV   #Perforing grid search

param_test1 = {
 'max_depth':np.arange(4,8,2),
 'min_child_weight': np.arange(1,6,2)
}

gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140,
gamma=0, subsample=0.8,colsample_bytree=0.8,
objective= 'multi:softmax', num_class=10, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='roc_auc',n_jobs=8,iid=False, cv=5)
gsearch1.fit(X_train, y_train)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
exit()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)  # label可以不要，此处需要是为了测试效果


dtarget = xgb.DMatrix(test_weight)
param = {'max_depth': 12, 'eta':0.05, 'eval_metric':'merror', 
             'max_delta_step': 0, 'subsample': 1, 'alpha': 1, 'lambda': 1, 'scale_pos_weight': 1,
    'n_estimators': 628, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 'base_score': 0.5,
    'silent': 1, 'objective':'multi:softmax', 'num_class':10, 'seed': 1857}  # 参数
evallist  = [(dtrain,'train'), (dtest,'test')]  # 这步可以不要，用于测试效果
num_round = 1000  # 循环次数
bst = xgb.train(param, dtrain, num_round, evallist)
preds = bst.predict(dtarget)

exit()
# In[31]:


submit = pd.DataFrame(columns=['content_id'])
submit['content_id']=target['content_id']
submit['subject'] = pd.Series(le.inverse_transform(preds.astype(int)), index=target.index)
submit['sentiment_value'] = 0
submit['sentiment_word'] = ""
submit.to_csv("submit.csv", index=False)


# In[ ]:






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

vectorizer = CountVectorizer(ngram_range=(1, 2),min_df=1)
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train["content_words"]))
weight = tfidf.toarray()
print(weight)
test_tfidf = tfidftransformer.transform(vectorizer.transform(target["content_words"]))
test_weight = test_tfidf.toarray()
print(test_weight.shape)


# In[21]:


from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(weight, train['subject_no'].values, test_size=0.1, random_state=42)#
X_train, y_train = weight, train['subject_no'].values


import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=12, n_estimators=682, learning_rate=0.05,
gamma=0, max_delta_step=0, subsample=1, colsample_bytree=0.9, colsample_bylevel=0.9,
                            reg_alpha=1, reg_lambda=1, scale_pos_weight=1,
                            base_score=0.5, seed=1750,
                        objective='multi:softmax', num_class=10)

# silent=True,
gbm.fit(X_train, y_train)
dtarget = xgb.DMatrix(test_weight)
preds = gbm.predict(dtarget)


# In[31]:


submit = pd.DataFrame(columns=['content_id'])
submit['content_id']=target['content_id']
submit['subject'] = pd.Series(le.inverse_transform(preds.astype(int)), index=target.index)
submit['sentiment_value'] = 0
submit['sentiment_word'] = ""
submit.to_csv("submit_ngfit_1010.csv", index=False)


# In[ ]:





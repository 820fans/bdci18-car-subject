

import argparse
import config
import logging
import os
import util

from keras import backend as K
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from capsule import *
import jieba

# not enable in windows
# jieba.enable_parallel(4)

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] (%(threadName)s) <%(processName)s> %(asctime)s %(message)s')
logger = logging.getLogger(__name__)

K.clear_session()
remove_stop_words = False

logger.info("load embedding vectors start.")
# load Glove Vectors
embeddings_index = {}
EMBEDDING_DIM = 300
with open(config.embedding_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
        values = line.split()
        words = values[:-EMBEDDING_DIM]
        word = ''.join(words)
        try:
            coefs = np.asarray(values[-EMBEDDING_DIM:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            pass
logger.info('load complete!Found %s word vectors.' % len(embeddings_index))

logger.info("load train/test/stopwords data.")
train_df = pd.read_csv(config.train_data_path, encoding='utf-8')
test_df = pd.read_csv(config.test_data_path, encoding='utf-8')
stop_words = util.load_stopwords(config.stop_words_path)
logger.info("load complete.")


train_df['label'] = train_df['subject'].str.cat(train_df['sentiment_value'].astype(str))

import tensorflow
from tensorflow.tensor.basic import as_tensor_variable

def my_f1(y_true, y_pred):
    # proba to label
    ypred = as_tensor_variable(y_pred)
    y_true = as_tensor_variable(y_true)
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = 0
    tp, fp, fn = 0, 0, 0
    for i in range(y_true.shape[0]):
        sum1, sum2 = y_true[i].sum(), y_pred[i].sum()
        if sum1 > sum2:
            # miss pred num
            fn += (sum1 - sum2)
        elif sum1 < sum2:
            fp += (sum2 - sum1)
        flag = True
        for j in range(y_true.shape[1]):
            if y_true[i, j] != y_pred[i, j]:
                flag = False
                break
        if flag:
            tp += 1
        else:
            fp += 1
    # precision
    p = tp / (tp + fp + K.epsilon())
    # recall
    r = tp / (tp + fn + K.epsilon())
    return 2 * p * r / (p + r + K.epsilon())


if remove_stop_words:
    train_df['content'] = train_df.content.map(
        lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
    test_df['content'] = test_df.content.map(
        lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
else:
    train_df['content'] = train_df.content.map(lambda x: ''.join(x.strip().split()))
    test_df['content'] = test_df.content.map(lambda x: ''.join(x.strip().split()))

train_dict = {}
for ind, row in train_df.iterrows():
    content, label = row['content'], row['label']
    if train_dict.get(content) is None:
        train_dict[content] = {label}
    else:
        train_dict[content].add(label)

conts = []
labels = []
for k, v in train_dict.items():
    conts.append(k)
    labels.append(v)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(labels)

content_list = [jieba.lcut(str(c), cut_all=True) for c in conts]

test_content_list = [jieba.lcut(c, cut_all=True) for c in test_df.content.astype(str).values]
word_set = set([word for row in list(content_list) + list(test_content_list) for word in row])
print(len(word_set))
word2index = {w: i + 1 for i, w in enumerate(word_set)}
seqs = [[word2index[w] for w in l] for l in content_list]
seqs_dev = [[word2index[w] for w in l] for l in test_content_list]

embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_DIM))
for word, i in word2index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = len(word_set) + 1


def get_padding_data(maxlen=100):
    x_train = sequence.pad_sequences(seqs, maxlen=maxlen)
    x_dev = sequence.pad_sequences(seqs_dev, maxlen=maxlen)
    return x_train, x_dev


def get_capsule_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(30, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[my_f1])
    return model


maxlen = 100
X_train, X_dev = get_padding_data(maxlen)
print(X_train.shape, X_dev.shape, y_train.shape)

# train model and find params
# model = get_capsule_model()
# batch_size = 30
# epochs = 50
# file_path = "weights_base.best.hdf5"
# checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
# callbacks_list = [checkpoint, early]  # early
# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

first_model_results = []
for i in range(5):
    model = get_capsule_model()
    model.fit(X_train, y_train, batch_size=64, epochs=15)
    first_model_results.append(model.predict(X_dev, batch_size=1024))
pred4 = np.average(first_model_results, axis=0)

tmp = [[i for i in row] for row in pred4]

for i, v in enumerate(tmp):
    if max(v) < 0.5:
        max_val = max(v)
        tmp[i] = [1 if j == max_val else 0 for j in v]
    else:
        tmp[i] = [int(round(j)) for j in v]

tmp = np.asanyarray(tmp)
res = mlb.inverse_transform(tmp)

cids = []
subjs = []
sent_vals = []
for c, r in zip(test_df.content_id, res):
    for t in r:
        if '-' in t:
            sent_val = -1
            subj = t[:-2]
        else:
            sent_val = int(t[-1])
            subj = t[:-1]
        cids.append(c)
        subjs.append(subj)
        sent_vals.append(sent_val)

res_df = pd.DataFrame({'content_id': cids, 'subject': subjs, 'sentiment_value': sent_vals,
                       'sentiment_word': ['一般' for i in range(len(cids))]})

columns = ['content_id', 'subject', 'sentiment_value', 'sentiment_word']
res_df = res_df.reindex(columns=columns)
res_df.to_csv('submit_capsule_word_my_f1.csv', encoding='utf-8', index=False)



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import pandas as pd

from sklearn.metrics import f1_score

stop_words = []


def load_stopwords(file_name):
    # load stopwords
    with open(file_name) as f:
        stop_words = set([l.strip() for l in f])
        return stop_words

def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents):
    contents_segs = list()
    for content in contents:
        rcontent = content.replace("\r\n", " ").replace("\n", " ")
        segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
        contents_segs.append(" ".join(segs))
    return contents_segs


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0, -1, -2], average='macro')

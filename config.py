#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

local_embedding_path = os.path.abspath('/') + "media/yida/yida_save/Project/Python/Data/embedding/sgns.baidubaike.bigram-char"
embedding_path = "/public/liangy/Data/embedding/sgns.baidubaike.bigram-char"
stop_words_path = "hlp_stopwords.txt"
data_path = os.path.abspath('.') + "/data/"
model_path = data_path + "model/"
train_data_path = data_path + "train.csv"
validate_data_path = data_path + "valid.csv"
test_data_path = data_path + "test.csv"
test_data_predict_output_path = data_path + "test_predict.csv"

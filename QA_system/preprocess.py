# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午4:51
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : preprocess.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np


def load_glove(filename):
    vocabulary = []
    embedding = []
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocabulary.append(row[0])
        embedding.append(row[1:])
    print('Loaded GloVe!')

    embedding = np.asarray(embedding)

    return vocabulary, embedding

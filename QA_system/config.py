# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午7:30
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : config.py
# @Software: PyCharm Community Edition


import os


BATCH_SIZE = 64
TRANING_EPOCH = 32

DATA_DIR = os.environ['DATA_DIR'] + '/'
# DATA_DIR = 'dataset/'

DATA_PATH = DATA_DIR + ""
TOKEN_OF_OUT_OF_VOCABULARY = "--OOV--"
CKP_PATH = DATA_DIR + ""
LOG_PATH = DATA_DIR + ""

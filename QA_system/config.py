# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午7:30
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : config.py
# @Software: PyCharm Community Edition


import os


BATCH_SIZE = 128
TRANING_EPOCH = 32


DATA_DIR = os.environ['DATA_DIR']
LOG_PATH = os.environ['LOG_DIR']
CKPT_PATH = os.environ['CHECKPOINT_DIR']
RESULT_PATH = os.path.join(os.environ['RESULT_DIR'], 'model')

'''
DATA_DIR = 'dataset/'
LOG_PATH = DATA_DIR
CKPT_PATH = DATA_DIR
RESULT_PATH = DATA_DIR
'''

DATA_PATH = DATA_DIR
TOKEN_OF_OUT_OF_VOCABULARY = "--OOV--"

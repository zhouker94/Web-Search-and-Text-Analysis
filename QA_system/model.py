# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午4:32
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : model.py
# @Software: PyCharm Community Edition


import tensorflow as tf
import constants as const
import layers


class Model(object):
    def __init__(self):
        self.context = tf.placeholder(tf.int32,
                                      [None, None, const.WORD_EMBEDDING_DIM],
                                      "context_input")
        self.question = tf.placeholder(tf.int32,
                                       [None, None, const.WORD_EMBEDDING_DIM],
                                       "question_input")

        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')

        self.emb_mat = tf.Variable(tf.constant(0.0, shape=[const.VOCABULARY_SIZE, const.WORD_EMBEDDING_DIM]),
                                   trainable=False, name="embedding_matrix")
        self.embedding_placeholder = tf.placeholder(tf.float32, [const.VOCABULARY_SIZE, const.WORD_EMBEDDING_DIM])
        self.embedding_init = self.emb_mat.assign(self.embedding_placeholder)

        self._build_model()

    def _build_model(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            c_emb = tf.reshape(tensor=tf.nn.embedding_lookup(self.emb_mat, self.context),
                               shape=[-1, -1, const.WORD_EMBEDDING_DIM])
            q_emb = tf.reshape(tensor=tf.nn.embedding_lookup(self.emb_mat, self.question),
                               shape=[-1, -1, const.WORD_EMBEDDING_DIM])

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = layers.encoder_block(c_emb, const.NUM_CONV_LAYERS, 7, self.dropout_keep_prob)
            q = layers.encoder_block(q_emb, const.NUM_CONV_LAYERS, 7, self.dropout_keep_prob)


if __name__ == '__main__':
    Model()

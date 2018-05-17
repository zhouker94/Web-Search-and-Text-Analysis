# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午4:32
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : model.py
# @Software: PyCharm Community Edition


import tensorflow as tf
import constants as const
import layers
import numpy as np


class Model(object):
    def __init__(self, emb_mat):
        self.context_input = tf.placeholder(tf.int32, shape=[const.BATCH_SIZE, None], name="context_input")
        self.question_input = tf.placeholder(tf.int32, shape=[const.BATCH_SIZE, None], name="question_input")

        self.label_start = tf.placeholder(tf.float32,
                                          [None, None, 1],
                                          "start_label")
        self.label_end = tf.placeholder(tf.float32,
                                        [None, None, 1],
                                        "end_label")

        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')

        self.emb_mat = tf.Variable(emb_mat, trainable=False, dtype=tf.float32)
        self.c = tf.nn.embedding_lookup(params=self.emb_mat, ids=self.context_input)
        self.q = tf.nn.embedding_lookup(params=self.emb_mat, ids=self.question_input)
        self.opm = None

        self._build_model()

    def _build_model(self):
        pass


"""
class QANetModel(Model):
    def __init__(self):
        super().__init__()

    def _build_model(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            c_emb = tf.reshape(tensor=tf.nn.embedding_lookup(self.emb_mat, self.context),
                               shape=[-1, -1, const.WORD_EMBEDDING_DIM])
            q_emb = tf.reshape(tensor=tf.nn.embedding_lookup(self.emb_mat, self.question),
                               shape=[-1, -1, const.WORD_EMBEDDING_DIM])

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = layers.encoder_block(c_emb, const.NUM_CONV_LAYERS, 7)
            q = layers.encoder_block(q_emb, const.NUM_CONV_LAYERS, 7)
"""


class EncoderDecoderModel(Model):
    def __init__(self, emb_mat):
        super().__init__(emb_mat)

    def _build_model(self):
        with tf.variable_scope("context_encoder_block"):
            encode_c = layers.rnn_encoder_block(self.c, self.dropout_keep_prob, False, "ec")
            print(encode_c.shape)
            encode_c_unstuck = tf.unstack(encode_c, axis=0)[0]

        with tf.variable_scope("question_encoder_block"):
            encode_q = layers.rnn_encoder_block(self.q, self.dropout_keep_prob, True, "eq")

        with tf.variable_scope("question_context_attention"):
            similarity = tf.map_fn(lambda x: tf.multiply(x, encode_c_unstuck), encode_q)
            norm_similarity = tf.nn.softmax(similarity, axis=1)

        with tf.variable_scope("qc_encoder_block"):
            qc_encode_layer_1 = layers.rnn_encoder_block(norm_similarity, self.dropout_keep_prob, False, "ec1")
            qc_encode_layer_2 = layers.rnn_encoder_block(qc_encode_layer_1, self.dropout_keep_prob, False, "ec2")
            qc_encode_layer_3 = layers.rnn_encoder_block(qc_encode_layer_2, self.dropout_keep_prob, False, "ec3")

            fc_1 = tf.contrib.layers.fully_connected(tf.concat([qc_encode_layer_1, qc_encode_layer_2], axis=2),
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                     activation_fn=None
                                                     )

            fc_2 = tf.contrib.layers.fully_connected(tf.concat([qc_encode_layer_2, qc_encode_layer_3], axis=2),
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                     activation_fn=None
                                                     )

        with tf.variable_scope("loss"):
            cross_entropy_1 = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_start, logits=fc_1))
            cross_entropy_2 = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_end, logits=fc_2))

            with tf.name_scope('adam_optimizer'):
                self.opm = tf.train.AdamOptimizer(1e-4).minimize(tf.add(cross_entropy_1, cross_entropy_2),
                                                                 name="optimizer")


if __name__ == "__main__":
    EncoderDecoderModel(np.asarray([[4, 5, 6]]))

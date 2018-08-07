# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午4:32
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : models.py
# @Software: PyCharm Community Edition


import tensorflow as tf
from layers import Layers
import numpy as np


class Model(object):
    def __init__(self, emb_mat):
        with tf.variable_scope("inputs"):
            self.context_input = tf.placeholder(tf.int32, shape=[None, None], name="context_input")
            self.question_input = tf.placeholder(tf.int32, shape=[None, None], name="question_input")

            self.label_start = tf.placeholder(tf.int32, [None], "start_label")
            self.label_end = tf.placeholder(tf.int32, [None], "end_label")

            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')

        with tf.variable_scope("embedding"):
            self.emb_mat = tf.Variable(emb_mat, trainable=False, dtype=tf.float32)
            self.c = tf.nn.embedding_lookup(params=self.emb_mat, ids=self.context_input)
            self.q = tf.nn.embedding_lookup(params=self.emb_mat, ids=self.question_input)

        self.opm = None

        self._build_model()

        self.merged = tf.summary.merge_all()

    def _build_model(self):
        pass


class RnnModel(Model):
    def __init__(self, emb_mat):
        super().__init__(emb_mat)

    def _build_model(self):
        with tf.variable_scope("context_encoder_block"):
            encode_c = Layers.rnn_block(self.c, self.dropout_keep_prob, "ec")

        with tf.variable_scope("question_encoder_block"):
            encode_q = Layers.rnn_block(self.q, self.dropout_keep_prob, "eq")

        with tf.variable_scope("question_context_coattention"):
            c_q_coattention = Layers.coattention(encode_c, encode_q)
            q_c_coattention = Layers.coattention(encode_q, encode_c)
            c_attention = Layers.coattention(c_q_coattention, q_c_coattention)

        with tf.variable_scope("qc_decode_block"):
            decode_c = Layers.rnn_block(c_attention, self.dropout_keep_prob, "decode_qc")
            # decode_qc_2 = Layers.rnn_block(c_attention, self.dropout_keep_prob, "decode_qc_2")

            fc_1 = tf.contrib.layers.fully_connected(decode_c,
                                                     256,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                                     activation_fn=tf.nn.relu
                                                     )
            
            fc_2 = tf.contrib.layers.fully_connected(fc_1,
                                                     256,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                                     activation_fn=tf.nn.relu
                                                     )


            output_1 = tf.contrib.layers.fully_connected(decode_c,
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                                     activation_fn=tf.nn.relu
                                                     )

            output_2 = tf.contrib.layers.fully_connected(decode_c,
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                                     activation_fn=tf.nn.relu
                                                     )

            output_1 = tf.squeeze(output_1, axis=-1)
            output_2 = tf.squeeze(output_2, axis=-1)

            self.output_layer_1 = tf.argmax(output_1, axis=-1)
            self.output_layer_2 = tf.argmax(output_2, axis=-1)

            tf.summary.histogram('logits_1', output_1)
            tf.summary.histogram('logits_2', output_2)

        with tf.variable_scope("loss"):
            cross_entropy_1 = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_1, labels=self.label_start)
            cross_entropy_2 = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_2, labels=self.label_end)

            tf.summary.histogram('cross_entropy_1', cross_entropy_1)
            tf.summary.histogram('cross_entropy_2', cross_entropy_2)

            self.loss = tf.reduce_mean(tf.add(cross_entropy_1, cross_entropy_2))
            tf.summary.scalar("training_Loss", self.loss)

            with tf.name_scope('adam_optimizer'):
                self.opm = tf.train.AdamOptimizer(1e-3).minimize(self.loss, name="optimizer")


"""
class CnnModel(Model):
    def __init__(self, emb_mat):
        super().__init__(emb_mat)

    def _build_model(self):
        with tf.variable_scope("context_encoder_block"):
            encode_c = Layers.cnn_block(self.c, self.dropout_keep_prob, "ec")

        with tf.variable_scope("question_encoder_block"):
            encode_q = Layers.cnn_block(self.q, self.dropout_keep_prob, "eq")

        # with tf.variable_scope("question_context_coattention"):


        with tf.variable_scope("qc_decode_block"):
            decode_qc_1 = Layers.cnn_block(c_attention, self.dropout_keep_prob, "decode_qc_1")
            decode_qc_2 = Layers.cnn_block(c_attention, self.dropout_keep_prob, "decode_qc_2")
            
            fc_1 = tf.contrib.layers.fully_connected(tf.concat([decode_qc_1, decode_qc_2], 2),
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                                     activation_fn=tf.nn.relu
                                                     )

            fc_2 = tf.contrib.layers.fully_connected(tf.concat([decode_qc_1, decode_qc_2], 2),
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                                     activation_fn=tf.nn.relu
                                                     )
            self.output_layer_1 = tf.squeeze(fc_1, axis=-1)
            self.output_layer_2 = tf.squeeze(fc_2, axis=-1)

        with tf.variable_scope("loss"):
            cross_entropy_1 = \
                tf.nn.softmax_cross_entropy_with_logits(labels=self.label_start, logits=self.output_layer_1)
            cross_entropy_2 = \
                tf.nn.softmax_cross_entropy_with_logits(labels=self.label_end, logits=self.output_layer_2)
            tf.summary.histogram('cross_entropy_1', cross_entropy_1)
            tf.summary.histogram('cross_entropy_2', cross_entropy_2)

            self.loss = tf.reduce_mean(tf.add(cross_entropy_1, cross_entropy_2))
            tf.summary.scalar("training_Loss", self.loss)

            with tf.name_scope('adam_optimizer'):
                self.opm = tf.train.AdamOptimizer(1e-3).minimize(self.loss, name="optimizer")
"""

if __name__ == "__main__":
    formed_input = np.zeros((2, 50))
    RnnModel(formed_input)

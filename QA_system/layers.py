# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:00
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


class Layers:
    @staticmethod
    def rnn_block(inputs, dropout_keep_prob, scope, compose=True):
        with tf.variable_scope(scope):
            norm_layer = tf.contrib.layers.layer_norm(inputs)

            gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell([Layers.dropout_wrapped_gru_cell(dropout_keep_prob)
                                                       for _ in range(2)])
            gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell([Layers.dropout_wrapped_gru_cell(dropout_keep_prob)
                                                       for _ in range(2)])

            encode_out, h = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_fw,
                                                            cell_bw=gru_cell_bw,
                                                            inputs=norm_layer,
                                                            dtype=tf.float32)
            if compose:
                encode_out = tf.concat(encode_out, 2)
                # shape [batch_size, word_length, encode_size]
                encode_out = Layers.self_attention(encode_out)

            return encode_out

    @staticmethod
    def self_attention(encoder_output):
        norm_layer = tf.contrib.layers.layer_norm(encoder_output)
        # (batch, words, ecode)
        W_1 = tf.layers.dense(norm_layer, 128, use_bias=False)
        W_2 = tf.layers.dense(norm_layer, 128, use_bias=False)

        # (batch, word, word)
        W_1_2 = tf.matmul(W_1, tf.transpose(W_2, [0, 2, 1]))
        return tf.matmul(tf.nn.softmax(W_1_2), norm_layer)

    @staticmethod
    def dropout_wrapped_gru_cell(in_keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(num_units=64, activation=tf.nn.relu)
        rnn_layer = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=in_keep_prob)
        return rnn_layer

    @staticmethod
    def coattention(encode_c, encode_q):
        variation_q = tf.transpose(encode_q, [0, 2, 1])
        L = tf.matmul(encode_c, variation_q)
        L_t = tf.transpose(L, [0, 2, 1])
        # normalize with respect to question
        a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
        # normalize with respect to context
        a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
        # summaries with respect to question, (batch_size, question+1, hidden_size)
        c_q = tf.matmul(a_q, encode_c)
        c_q_emb = tf.concat((variation_q, tf.transpose(c_q, [0, 2, 1])), 1)
        # summaries of previous attention with respect to context
        c_d = tf.matmul(c_q_emb, a_c, adjoint_b=True)
        # final coattention context, (batch_size, context+1, 3*hidden_size)
        co_att = tf.concat((encode_c, tf.transpose(c_d, [0, 2, 1])), 2)
        # shape [Batch, c, q]
        return co_att

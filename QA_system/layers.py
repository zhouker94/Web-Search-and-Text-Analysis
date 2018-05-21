# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:00
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


class Layers:
    @staticmethod
    def rnn_encoder_block(inputs, dropout_keep_prob, scope):
        with tf.variable_scope(scope):
            norm_layer = tf.contrib.layers.layer_norm(inputs)
            print(norm_layer.shape)

            gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell([Layers.dropout_wrapped_gru_cell(dropout_keep_prob)
                                                       for _ in range(2)])
            gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell([Layers.dropout_wrapped_gru_cell(dropout_keep_prob)
                                                       for _ in range(2)])

            encode_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_fw,
                                                            cell_bw=gru_cell_bw,
                                                            inputs=norm_layer,
                                                            dtype=tf.float32)
            encode_out = tf.concat(encode_out, 2)
            encode_out = Layers.self_attention(encode_out)

        # shape [batch_size, word_length, encode_size]
        return encode_out

    @staticmethod
    def self_attention(encoder_output):
        # (batch, words, ecode)
        W_1 = tf.layers.dense(encoder_output, 64, use_bias=False)
        W_2 = tf.layers.dense(encoder_output, 64, use_bias=False)

        # (batch, word, word)
        W_1_2 = tf.matmul(W_1, tf.transpose(W_2, [0, 2, 1]))
        return tf.matmul(tf.nn.softmax(W_1_2), encoder_output)

    @staticmethod
    def dropout_wrapped_gru_cell(in_keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(num_units=64, activation=tf.nn.sigmoid)
        rnn_layer = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=in_keep_prob)
        return rnn_layer

    @staticmethod
    def similarity(encode_c, encode_q):
        W_c = tf.layers.dense(encode_c, 128, use_bias=False)
        W_q = tf.layers.dense(encode_q, 128, use_bias=False)
        W_q_T = tf.transpose(W_q, [0, 2, 1])
        similarity = tf.matmul(W_c, W_q_T)
        # shape [Batch, c, q]
        return similarity

# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:00
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


class Layers:
    """
    @staticmethod
    def cnn_block(inputs, dropout_keep_prob, scope):
        with tf.variable_scope(scope):
            norm_layer = tf.contrib.layers.layer_norm(inputs)
            print(norm_layer.shape)
            conv_layer_1 = Layers.conv1d_layer(norm_layer)
            conv_layer_2 = Layers.conv1d_layer(conv_layer_1)
            conv_layer_3 = Layers.conv1d_layer(conv_layer_2)
        return conv_layer_3
    """

    @staticmethod
    def rnn_block(inputs, dropout_keep_prob, scope, is_encode=True, is_compose=True):
        with tf.variable_scope(scope):
            norm_layer = tf.contrib.layers.layer_norm(inputs)

            gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell([Layers.dropout_wrapped_gru_cell(dropout_keep_prob)
                                                       for _ in range(2)])
            gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell([Layers.dropout_wrapped_gru_cell(dropout_keep_prob)
                                                       for _ in range(2)])

            encode_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_fw,
                                                            cell_bw=gru_cell_bw,
                                                            inputs=norm_layer,
                                                            dtype=tf.float32)

            if not is_compose:
                return encode_out[0], encode_out[1]

            # shape [batch_size, word_length, encode_size]
            encode_out = tf.concat(list(encode_out), axis=2)

            if is_encode:
                with tf.variable_scope("self_attention"):
                    self_attention = Layers.self_attention(encode_out)
                    gate = tf.layers.dense(self_attention, self_attention.shape[-1],
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                                           activation=tf.nn.sigmoid, use_bias=False)
                    encode_out = gate * self_attention
            return encode_out

    @staticmethod
    def self_attention(encoder_output):
        """
        # (batch, words, encode) using bi-linear
        W_1 = tf.layers.dense(encoder_output, 128, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001), activation=tf.nn.relu)
        W_2 = tf.layers.dense(encoder_output, 128, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001), activation=tf.nn.relu)
        # (batch, word, word)
        W_1_2 = tf.matmul(W_1, tf.transpose(W_2, [0, 2, 1]))
        """
        norm_layer = tf.contrib.layers.layer_norm(encoder_output)
        W_1_2 = tf.matmul(norm_layer, tf.transpose(norm_layer, [0, 2, 1]))
        return tf.matmul(tf.nn.softmax(W_1_2), norm_layer)

    @staticmethod
    def dropout_wrapped_gru_cell(in_keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(num_units=64, activation=tf.nn.relu)
        rnn_layer = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=in_keep_prob)
        return rnn_layer

    @staticmethod
    def coattention(encode_c, encode_q):
        # (batch_size, hidden_size，question)
        variation_q = tf.transpose(encode_q, [0, 2, 1])
        # [batch, c length, q length]
        L = tf.matmul(encode_c, variation_q)
        L_t = tf.transpose(L, [0, 2, 1])
        # normalize with respect to question
        a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
        # normalize with respect to context
        a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
        # summaries with respect to question, (batch_size, question, hidden_size)
        c_q = tf.matmul(a_q, encode_c)
        c_q_emb = tf.concat((variation_q, tf.transpose(c_q, [0, 2, 1])), 1)
        # summaries of previous attention with respect to context
        c_d = tf.matmul(c_q_emb, a_c, adjoint_b=True)
        # coattention context [batch_size, context+1, 3*hidden_size]
        co_att = tf.concat((encode_c, tf.transpose(c_d, [0, 2, 1])), 2)
        return co_att

    """
    @staticmethod
    def conv1d_layer(inputs):
        weight = tf.Variable(tf.truncated_normal(
            [4, int(inputs.shape[2]), 128]))
        bias = tf.Variable(tf.zeros(128))
        conv_layer = tf.nn.conv1d(inputs, weight, stride=1, padding='SAME')
        conv_layer = tf.nn.bias_add(conv_layer, bias)
        conv_layer = tf.nn.relu(conv_layer)
        return conv_layer
    """

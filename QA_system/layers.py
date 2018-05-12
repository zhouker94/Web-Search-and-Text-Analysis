# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:00
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


def encoder_block(inputs, num_conv_layers, kernel_size, scope, num_filters=128):
    with tf.variable_scope(scope):
        with tf.variable_scope("conv_block"):
            curr_inputs = pe
            for i in range(num_conv_layers):
                res = curr_inputs
                curr_inputs = tf.contrib.layers.layer_norm(curr_inputs)
                curr_inputs = tf.layers.conv1d(curr_inputs, num_filters, kernel_size, padding="SAME",
                                               activation=tf.nn.relu)
                curr_inputs = tf.add(curr_inputs, res)

        with tf.variable_scope("self_attention_block"):
            init = curr_inputs
            layer_norm_1 = tf.contrib.layers.layer_norm(curr_inputs)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=128)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                layer_norm_1, attention_mechanism, attention_layer_size=128 / 2)
            curr_inputs = tf.add(attn_cell, init)

        with tf.variable_scope("self_feedfoward_block"):
            layer_norm_2 = tf.contrib.layers.layer_norm(curr_inputs)
            fc = tf.contrib.layers.fully_connected(layer_norm_2,
                                                   128,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
            outputs = tf.nn.dropout(fc, keep_prob)

        return outputs


def rnn_encoder_block(inputs, dropout_keep_prob, is_q, scope):
    with tf.variable_scope(scope):
        norm_layer = tf.contrib.layers.layer_norm(inputs)

        gru_cell = tf.contrib.rnn.GRUCell(num_units=128, activation=tf.nn.relu)
        rnn_layer = tf.contrib.rnn.DropoutWrapper(gru_cell,
                                                  input_keep_prob=dropout_keep_prob,
                                                  output_keep_prob=dropout_keep_prob,
                                                  state_keep_prob=dropout_keep_prob,
                                                  )
        attention_cell = tf.contrib.rnn.AttentionCellWrapper(rnn_layer, 7)
        encode_out, states = tf.nn.dynamic_rnn(cell=attention_cell, inputs=norm_layer, dtype=tf.float32)

    if is_q:
        # shape [batch_size, cell_state_size]
        outputs = states[0]
    else:
        # shape [batch_size, context_length, cell_state_size]
        outputs = encode_out

    return outputs

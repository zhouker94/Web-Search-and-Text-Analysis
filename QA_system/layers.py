# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:00
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


def encoder_block(inputs, num_conv_layers, kernel_size, keep_prob, num_filters=128, scope="encoder_block"):
    with tf.variable_scope(scope):

        with tf.variable_scope("position_encoding"):
            pe = pos_encoding(inputs)

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


def pos_encoding(x):
    pass
'''
    (_, d, l) = x.size()
    pos = torch.arange(l).repeat(d, 1)
    tmp1 = tf.multiply(pos, freqs)
    tmp2 = tf.add(tmp1, phases)
    pos_enc = tf.sin(tmp2)
    out = tf.sin(pos_enc) + x
    return out
'''

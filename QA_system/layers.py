# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:00
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


def encoder_block(inputs, num_conv_layers, kernel_size, num_filters=128, scope="encoder_block"):
    with tf.variable_scope(scope):

        with tf.variable_scope("conv_block"):
            curr_inputs = inputs
            for i in range(num_conv_layers):
                init = curr_inputs
                curr_inputs = tf.contrib.layers.layer_norm(curr_inputs)
                curr_inputs = tf.contrib.layers.conv2d(curr_inputs, num_filters, kernel_size)
                curr_inputs = tf.add(curr_inputs, init)

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
        outputs = fc
        return outputs

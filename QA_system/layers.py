# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:00
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


class Layers:

    @staticmethod
    def birnn_layer(input_, num_units=64, dropout_keep_prob=1.0, num_layers=2):

        gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            [
                Layers.dropout_wrapped_gru_cell(dropout_keep_prob, num_units)
                for _ in range(num_layers)
            ]
        )

        gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [
                Layers.dropout_wrapped_gru_cell(dropout_keep_prob, num_units)
                for _ in range(num_layers)
            ]
        )

        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_fw,
                                                    cell_bw=gru_cell_bw,
                                                    inputs=input_,
                                                    dtype=tf.float32)
        return output

    @staticmethod
    def rnn_layer(input_, num_units=128, dropout_keep_prob=1.0, num_layers=2):

        gru_cell = tf.nn.rnn_cell.MultiRNNCell(
            [
                Layers.dropout_wrapped_gru_cell(dropout_keep_prob, num_units)
                for _ in range(num_layers)
            ]
        )

        output, _ = tf.nn.dynamic_rnn(cell=gru_cell,
                                      inputs=input_,
                                      dtype=tf.float32)
        return output

    @staticmethod
    def dropout_wrapped_gru_cell(in_keep_prob, num_units):

        gru_cell = tf.contrib.rnn.GRUCell(
            num_units=num_units,
            activation=tf.nn.relu
        )

        rnn_layer = tf.contrib.rnn.DropoutWrapper(
            gru_cell,
            input_keep_prob=in_keep_prob
        )

        return rnn_layer

    @staticmethod
    def add_dense_layer(input_, output_shape, drop_keep_prob, activation=tf.nn.relu, use_bias=True):

        output = input_
        for n in output_shape:
            output = tf.layers.dense(
                output,
                n,
                activation=activation,
                use_bias=use_bias
            )
            output = tf.nn.dropout(output, drop_keep_prob)
        return output

    @staticmethod
    def dot_attention(inputs, memory, hidden, keep_prob=1.0, activation=tf.nn.relu):

        inputs_ = Layers.add_dense_layer(
            inputs,
            hidden,
            keep_prob,
            activation=activation,
            use_bias=False
        )

        memory_ = Layers.add_dense_layer(
            memory,
            hidden,
            keep_prob,
            activation=activation,
            use_bias=False
        )

        outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1]))
        logits = tf.nn.softmax(outputs)
        outputs = tf.matmul(logits, memory)
        result = tf.concat([inputs, outputs], axis=-1)

        gate = Layers.add_dense_layer(
            result,
            [result.shape[-1]],
            keep_prob,
            activation=tf.nn.sigmoid,
            use_bias=False
        )

        return result * gate

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
        # summaries with respect to question, (batch_size, question,
        # hidden_size)
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

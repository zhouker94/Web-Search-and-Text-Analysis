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
        self.context = tf.placeholder(tf.float32,
                                      [1, None, const.WORD_EMBEDDING_DIM],
                                      "context_input")
        self.question = tf.placeholder(tf.float32,
                                       [None, None, const.WORD_EMBEDDING_DIM],
                                       "question_input")
        self.label_start = tf.placeholder(tf.float32,
                                          [None, None, 1],
                                          "start_label")
        self.label_end = tf.placeholder(tf.float32,
                                        [None, None, 1],
                                        "end_label")

        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')

        self._build_model()

    def _build_model(self):
        pass


class QANetModel(Model):
    def __init__(self):
        super().__init__()

    def _build_model(self):
        """
        with tf.variable_scope("Input_Embedding_Layer"):
            c_emb = tf.reshape(tf.nn.embedding_lookup(self.emb_mat, self.context))
            q_emb = tf.reshape(tf.nn.embedding_lookup(self.emb_mat, self.question))

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = layers.encoder_block(c_emb, const.NUM_CONV_LAYERS, 7)
            q = layers.encoder_block(q_emb, const.NUM_CONV_LAYERS, 7)
        """


class EncoderDecoderModel(Model):
    def __init__(self):
        super().__init__()

    def _build_model(self):
        with tf.variable_scope("context_encoder_block"):
            encode_c = layers.rnn_encoder_block(self.context, self.dropout_keep_prob, False, "ecc")
            encode_c_unstuck = tf.unstack(encode_c, axis=0)[0]

        with tf.variable_scope("question_encoder_block"):
            encode_q = layers.rnn_encoder_block(self.question, self.dropout_keep_prob, True, "ecq")

        with tf.variable_scope("question_context_attention"):
            similarity = tf.map_fn(lambda x: tf.multiply(x, encode_c_unstuck), encode_q)
            norm_similarity = tf.nn.softmax(similarity, axis=1)

        with tf.variable_scope("qc_encoder_block"):
            qc_encode_layer_1 = layers.rnn_encoder_block(norm_similarity, self.dropout_keep_prob, False, "ec1")
            qc_encode_layer_2 = layers.rnn_encoder_block(qc_encode_layer_1, self.dropout_keep_prob, False, "ec2")
            qc_encode_layer_3 = layers.rnn_encoder_block(qc_encode_layer_2, self.dropout_keep_prob, False, "ec3")

            fc_1 = tf.contrib.layers.fully_connected(tf.concat([qc_encode_layer_1, qc_encode_layer_2], axis=1),
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

            fc_2 = tf.contrib.layers.fully_connected(tf.concat([qc_encode_layer_2, qc_encode_layer_3], axis=1),
                                                     1,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

        with tf.variable_scope("loss"):
            cross_entropy_1 = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_start, logits=fc_1))
            cross_entropy_2 = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_end, logits=fc_2))

            with tf.name_scope('adam_optimizer'):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(tf.add(cross_entropy_1, cross_entropy_2))


if __name__ == "__main__":
    EncoderDecoderModel()

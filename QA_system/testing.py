import models as model
import tensorflow as tf
import numpy as np
import pickle
import constants as const


with open("testing_data.pickle", "rb") as input_file:
    testing_data = pickle.load(input_file)

with open("vocabulary.pickle", "rb") as input_file:
    voc = pickle.load(input_file)

emb_mat = np.load("word_embedding_matrix.npy")

edm = model.EncoderDecoderModel(emb_mat)


def find_max_length(lst):
    length = max((len(e) for e in lst))
    return length


def convert_word_to_embedding_index(word, voc):
    if word in voc:
        return voc[word]
    else:
        return 0


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "model/rnn_model")
    print("load sucessfully")

    batch_i = 0
    start_list = []
    end_list = []
    print(len(testing_data))
    while batch_i < len(testing_data):
        start = batch_i

        end = batch_i + const.BATCH_SIZE

        if end > len(testing_data):
            end = len(testing_data)

        q_list = []
        c_list = []

        for ins in testing_data[start: end]:
            q_list.append(list(map(lambda x: convert_word_to_embedding_index(x, voc), ins['question'])))
            c_list.append(list(map(lambda x: convert_word_to_embedding_index(x, voc), ins['context'])))

        # padding to a matrix by '0' on axis 1
        while len(q_list) != const.BATCH_SIZE and len(c_list) != const.BATCH_SIZE:
            q_list.append([0])
            c_list.append([0])

        # padding to a matrix by '0' on axis 2
        max_q = find_max_length(q_list)
        for i in q_list:
            i.extend([0] * (max_q - len(i)))
        batch_q = np.asarray(q_list)

        max_c = find_max_length(c_list)
        for i in c_list:
            i.extend([0] * (max_c - len(i)))
        batch_c = np.asarray(c_list)

        start_point = tf.argmax(edm.fc_1, axis=1)
        end_point = tf.argmax(edm.fc_2, axis=1)

        c_s, c_e = sess.run([edm.fc_1, edm.fc_2], feed_dict={edm.context_input: batch_c,
                                                             edm.question_input: batch_q,
                                                             edm.dropout_keep_prob: 1.0
                                                             })

        print(c_s, c_e)
        print(c_s.shape, c_e.shape)
        start_list.extend(c_s)
        end_list.extend(c_e)
        batch_i += const.BATCH_SIZE

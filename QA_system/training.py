# coding: utf-8

import tensorflow as tf
import numpy as np
import pickle
import models
import constants as const
import random


with open("/mnt/training_data.pickle", "rb") as input_file:
    # training data is list of dictionary
    training_data = pickle.load(input_file)

emb_mat = np.load("/mnt/new_word_embedding_matrix.npy")
rm = models.RnnModel(emb_mat)

with open("/mnt/new_vocabulary.pickle", "rb") as input_file:
    voc = pickle.load(input_file)

with open("devel_data.pickle", "rb") as input_file:
    # training data is list of dictionary
    devel_data = pickle.load(input_file)


# this is for padding list of list words to batch matrix
def find_max_length(lst):
    length = max((len(e) for e in lst))
    return length


def convert_word_to_embedding_index(word, voc):
    if word in voc:
        return voc[word]
    else:
        return random.randint(0, len(voc) - 1)

    
with tf.Session() as sess:
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('model/train', sess.graph)

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "model/rnn")
    global_step = 0

    for epoch in range(20):
        batch_i = 0
        np.random.shuffle(training_data)
        while batch_i < len(training_data):
            start = batch_i
            end = batch_i + const.BATCH_SIZE

            batch_data = training_data[start: end]
            q_list = []
            c_list = []
            s_list = []
            e_list = []
            for ins in batch_data:
                q_list.append(list(map(lambda x: convert_word_to_embedding_index(x, voc), ins['question'])))
                c_list.append(list(map(lambda x: convert_word_to_embedding_index(x, voc), ins['context'])))
                s_list.append(list(ins['start']))
                e_list.append(list(ins['end']))

            # padding to a matrix by '0'
            max_q = find_max_length(q_list)
            for i in q_list:
                i.extend([0] * (max_q - len(i)))
            batch_q = np.asarray(q_list)

            max_c = find_max_length(c_list)
            for i in c_list:
                i.extend([0] * (max_c - len(i)))
            batch_c = np.asarray(c_list)

            max_s = find_max_length(s_list)
            for i in s_list:
                i.extend([0] * (max_s - len(i)))
            batch_s = np.asarray(s_list)

            max_e = find_max_length(e_list)
            for i in e_list:
                i.extend([0] * (max_e - len(i)))
            batch_e = np.asarray(e_list)

            print(batch_q.shape, batch_e.shape, batch_c.shape, batch_s.shape)
            _, loss, summaries = sess.run([rm.opm, rm.loss, rm.merged], feed_dict={rm.context_input: batch_c,
                                                                                   rm.question_input: batch_q,
                                                                                   rm.label_start: batch_s,
                                                                                   rm.label_end: batch_e,
                                                                                   rm.dropout_keep_prob: 0.8
                                                                                   })

            # every 10 global steps, sampling a batch to evaluate loss from devel set
            if not global_step % 10:
                # random batch
                dev_index = random.randint(0, len(devel_data) - 1)
                dev_batch_data = devel_data[dev_index: dev_index + const.BATCH_SIZE]
                
                dev_q_list = []
                dev_c_list = []
                dev_s_list = []
                dev_e_list = []
                for ins in dev_batch_data:
                    dev_q_list.append(list(map(lambda x: convert_word_to_embedding_index(x, voc), ins['question'])))
                    dev_c_list.append(list(map(lambda x: convert_word_to_embedding_index(x, voc), ins['context'])))
                    dev_s_list.append(list(ins['start']))
                    dev_e_list.append(list(ins['end']))

                # padding to a matrix by '0'
                dev_max_q = find_max_length(dev_q_list)
                for i in dev_q_list:
                    i.extend([0] * (dev_max_q - len(i)))
                dev_batch_q = np.asarray(dev_q_list)

                dev_max_c = find_max_length(dev_c_list)
                for i in dev_c_list:
                    i.extend([0] * (dev_max_c - len(i)))
                dev_batch_c = np.asarray(dev_c_list)

                dev_max_s = find_max_length(dev_s_list)
                for i in dev_s_list:
                    i.extend([0] * (dev_max_s - len(i)))
                dev_batch_s = np.asarray(dev_s_list)

                dev_max_e = find_max_length(dev_e_list)
                for i in dev_e_list:
                    i.extend([0] * (dev_max_e - len(i)))
                dev_batch_e = np.asarray(dev_e_list)
                
                loss = sess.run(rm.loss, feed_dict={rm.context_input: dev_batch_c,
                                          rm.question_input: dev_batch_q,
                                          rm.label_start: dev_batch_s,
                                          rm.label_end: dev_batch_e,
                                          rm.dropout_keep_prob: 1
                                          })
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = loss
                summary_value.tag = "evaluate_loss"
                writer.add_summary(summary, global_step)
            
            writer.add_summary(summaries, global_step)

            print("Epoch:", epoch, "loss:", loss)
            batch_i += const.BATCH_SIZE
            global_step += 1

        save_path = saver.save(sess, "model/rnn")
        print("Model saved in path: %s" % save_path)

# coding: utf-8

import tensorflow as tf
import numpy as np
import pickle
import models
import config
import random
import argparse
from tensorflow import keras


def load_dataset():
    with open(config.DATA_PATH + "trainset_context.pickle", "rb") as f:
        # training data is list of dictionary
        train_c = pickle.load(f)

    with open(config.DATA_PATH + "trainset_question.pickle", "rb") as f:
        # training data is list of dictionary
        train_q = pickle.load(f)

    with open(config.DATA_PATH + "devset_context.pickle", "rb") as f:
        # training data is list of dictionary
        dev_c = pickle.load(f)

    with open(config.DATA_PATH + "devset_question.pickle", "rb") as f:
        # training data is list of dictionary
        dev_q = pickle.load(f)

    emb_mat = np.load(config.DATA_PATH + "word_embedding_matrix.npy")

    with open(config.DATA_PATH + "vocabulary.pickle", "rb") as f:
        voc = pickle.load(f)

    return train_c, train_q, dev_c, dev_q, emb_mat, voc


def text_to_index(raw_text, vocb):
    word_seq = tf.keras.preprocessing.text.text_to_word_sequence(raw_text)

    index_list = []
    for w in word_seq:
        if w in vocb:
            index_list.append(vocb[w])
        else:
            index_list.append(vocb[config.TOKEN_OF_OUT_OF_VOCABULARY])

    return index_list


def generate_batch(batch_sample, voc, context):
    batch_q, batch_c, batch_s, batch_e = [], [], [], []
    for q in batch_sample:
        if not q["is_impossible"]:
            batch_q.append(text_to_index(q['question'], voc))
            batch_c.append(text_to_index(context[q['context_id']], voc))
            
            first_answer = q['answers'][0]
            answer_start = len(tf.keras.preprocessing.text.text_to_word_sequence(first_answer['answer_start']))
            answer_end =  answer_start + \
                len(tf.keras.preprocessing.text.text_to_word_sequence(first_answer['text'])) - 1
                
            batch_s.append(answer_start)
            batch_e.append(answer_end)

    batch_q = keras.preprocessing.sequence.pad_sequences(batch_q,
                                                         value=voc[
                                                             "<PAD>"],
                                                         padding='post',
                                                         maxlen=16)
    
    batch_c = keras.preprocessing.sequence.pad_sequences(batch_c,
                                                         value=voc[
                                                             "<PAD>"],
                                                         padding='post')

    return batch_q, batch_c, batch_s, batch_e


def train(warm_start):

    train_c, train_q, dev_c, dev_q, emb_mat, voc = load_dataset()

    with tf.Session() as sess:

        rm = models.RnnModel(emb_mat)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('model/train', sess.graph)

        if warm_start:
            ckpt = tf.train.get_checkpoint_state(config.CKP_PAHT)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Cannot restore model from {}".format(
                    ckpt.model_checkpoint_path))
                return
        else:
            sess.run(tf.global_variables_initializer())

        global_step = 0

        for epoch in range(config.TRANING_EPOCH):

            batch_counter = 0
            np.random.shuffle(train_q)
            while batch_counter < len(train_q):

                batch_sample = train_q[
                    batch_counter: batch_counter + config.BATCH_SIZE]

                batch_q, batch_c, batch_s, batch_e = generate_batch(
                    batch_sample, voc, train_c)

                _, loss, summaries = sess.run([rm.opm, rm.loss, rm.merged],
                                              feed_dict={rm.context_input: batch_c,
                                                         rm.question_input: batch_q,
                                                         rm.label_start: batch_s,
                                                         rm.label_end: batch_e,
                                                         rm.dropout_keep_prob: 0.8
                                                         })

                # every 16 global steps, sampling a batch to evaluate loss from
                # devel set
                if not global_step % 16:
                    # random batch
                    dev_index = random.randint(0, len(dev_q) - 1)
                    dev_batch_sample = dev_q[
                        dev_index: dev_index + config.BATCH_SIZE]

                    dev_batch_q, dev_batch_c, dev_batch_s, dev_batch_e = generate_batch(
                        dev_batch_sample, voc, dev_c)

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

                print(
                    " --- Epoch: {}, batch: {}, loss: {} --- ".format(epoch, batch_counter, loss))
                batch_counter += config.BATCH_SIZE
                global_step += 1

            save_path = saver.save(sess, "model/rnn")
            print("Model saved in path: {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_mode', type=bool, default=False,
                        help='Is warm start from existing checkpoint?')
    parsed_args = parser.parse_args()

    train(parsed_args.start_mode)

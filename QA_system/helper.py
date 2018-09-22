import pickle
import numpy as np
import config
from tensorflow import keras
import os

def load_dataset():
    with open(os.path.join(config.DATA_PATH, "trainset_context.pickle"), "rb") as f:
        # training data is list of dictionary
        train_c = pickle.load(f)

    with open(os.path.join(config.DATA_PATH, "trainset_question.pickle"), "rb") as f:
        # training data is list of dictionary
        train_q = pickle.load(f)

    '''
    with open(config.DATA_PATH + "devset_context.pickle", "rb") as f:
        # training data is list of dictionary
        dev_c = pickle.load(f)

    with open(config.DATA_PATH + "devset_question.pickle", "rb") as f:
        # training data is list of dictionary
        dev_q = pickle.load(f)
    '''

    emb_mat = np.load(os.path.join(config.DATA_PATH, "word_embedding_matrix.npy"))

    with open(os.path.join(config.DATA_PATH, "vocabulary.pickle"), "rb") as f:
        voc = pickle.load(f)

    return train_c, train_q, None, None, emb_mat, voc


def text_to_index(raw_text, vocb):
    word_seq = keras.preprocessing.text.text_to_word_sequence(raw_text)

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
        batch_q.append(text_to_index(q['question'], voc))
        batch_c.append(text_to_index(context[q['context_id']], voc))

        first_answer = q['answers'][0]
        answer_start = len(keras.preprocessing.text.text_to_word_sequence(first_answer['answer_start']))
        answer_end =  answer_start + \
            len(keras.preprocessing.text.text_to_word_sequence(first_answer['text'])) - 1

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

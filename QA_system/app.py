from flask import Flask
from flask import request, jsonify
import numpy as np
import tensorflow as tf
import config
import helper


app = Flask(__name__)
model = None
sess = None
voc = None


@app.route('/answer', methods=['GET'])
def predict():

    if request.method == 'GET':
        if not model:
            return jsonify({'status': False, 'response': 'Train a model first'})

        """
        Json formate:
        
        {
            "question_list":
            [
                {"question": "..."},
                {"question": "..."},
                
                ...

                {"question": "..."}
            ],
            "context_list":
            [
                {"context": "..."},
                {"context": "..."},

                ...

                {"context": "..."}
            ]
        }
        """

        json_ = request.get_json()
        
        q_list = json_["question_list"]
        c_list = json_["context_list"]
        a_list = []

        if len(q_list) > 256 or len(q_list) != len(c_list):
            return jsonify({'status': False, 'response': 'Exceed max limited or question & contet not match'})

        q_list = [helper.text_to_index(q["question"], voc) for q in q_list["question"]]
        c_list = [helper.text_to_index(c["context"], voc) for c in c_list["context"]]

        batch_q = keras.preprocessing.sequence.pad_sequences(q_list,
                                                             value=voc[
                                                                 "<PAD>"],
                                                             padding='post',
                                                             maxlen=16)
    
        batch_c = keras.preprocessing.sequence.pad_sequences(c_list,
                                                             value=voc[
                                                                "<PAD>"],
                                                             padding='post')

        start_p, end_p = sess.run([model.output_layer_1, model.output_layer_2],
                                  feed_dict={
                                                model.context_input: dev_batch_c,
                                                model.question_input: dev_batch_q
                                            })
        
        for i, (start, end) in enumerate(zip(start_p, end_p)):
            a_list.append(''.join(c_list[i][start: end + 1]))

        return jsonify({"status": True, "answer_list": a_list})


if __name__ == '__main__':
    
    # load GLOVE and Vocabulary

    emb_mat = np.load(config.DATA_PATH + "word_embedding_matrix.npy")

    with open(config.DATA_PATH + "vocabulary.pickle", "rb") as f:
        voc = pickle.load(f)

    # start session
    with tf.Session() as sess:

        model = models.RnnModel(emb_mat)
        saver = tf.train.Saver()
        
        # start model
        ckpt = tf.train.get_checkpoint_state(config.CKP_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Cannot restore model")
            return
        
        print('app start')

        app.run(host='0.0.0.0', port=5000, debug=True)


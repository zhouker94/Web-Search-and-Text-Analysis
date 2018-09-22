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
                {"id": 0, "text": "..."},
                {"id": 1, "text": "..."},
                
                ...

                {"id": n, "text": "..."}
            ],
            "document":
                "
                ...
                ...
                ...
                "
        }
        """

        json_ = request.get_json()

        question_list = json_["question_list"]
        context = json_["document"]
        answer_list = []

        if len(question_list) > 256:
            return jsonify({'status': False,
                            'response': 'Exceed max limited or question & context not match'})

        question_list = [helper.text_to_index(q["text"], voc)
                    for q in question_list]

        context = helper.text_to_index(context, voc)
        context_list = [context] * len(question_list)

        batch_q = keras.preprocessing.sequence.pad_sequences(question_list,
                                                             value=voc[
                                                                 "<PAD>"],
                                                             padding='post',
                                                             maxlen=16)

        batch_c = keras.preprocessing.sequence.pad_sequences(context_list,
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

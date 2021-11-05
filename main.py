from flask import Flask, request, jsonify,render_template
import os
from flask_cors import CORS, cross_origin
import argparse
# import tensorflow as tf
from TextCategorizer import TextCategorizer
from paths import Paths

paths = Paths()

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.json['data']
    text_cat = TextCategorizer(ARGS)
    pred_class = text_cat.categorize(data)
    return jsonify({"PREDICTED CLASS" : pred_class})



if __name__ ==  "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True)














    
    arg_parser = argparse.ArgumentParser()
    # RNN model hyperparameters
    arg_parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of every batch")
    arg_parser.add_argument("-s", "--seq_length", type=int, default=10, help="sequence length/unrollings")
    arg_parser.add_argument("-e", "--num_epochs", type=int, default=10001, help="number of epochs for training")

    arg_parser.add_argument("-u", "--hidden_units", type=int, default=128, help="number of units in the hidden layers")
    arg_parser.add_argument("-l", "--hidden_layers", type=int, default=1, help="number of hidden layers")
    arg_parser.add_argument("-d", "--dropout_prob", type=float, default=0.5, help="dropout probability while training")

    arg_parser.add_argument("-r", "--learning_rate", type=float, default=10.0, help="initial learning rate")

    # W2V hyperparameters
    arg_parser.add_argument("-we", "--w2v_embed_size", type=int, default=128, help="embedding dimension for Word2Vec")
    arg_parser.add_argument("-ww", "--w2v_window", type=int, default=5, help="skip window size for Word2Vec")

    # Running parameters
    arg_parser.add_argument("-rw", "--raw_data", help="Use unpreprocessed raw data", action="store_true")
    arg_parser.add_argument("-te", "--testing", help="flag to run the netwoek in testing mode", action="store_true")

    ARGS = arg_parser.parse_args()
    # Flask Server
    app.run(host='0.0.0.0', port=5000, debug=True)

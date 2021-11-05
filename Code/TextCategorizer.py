import os
import argparse
import tensorflow as tf
from nltk.corpus import stopwords

from word_embedder_gensim import WordVectorizer
from rnn_w2v import RNN_Model
from preprocess import preprocess_doc
from batch_generator import embedding_lookup
from paths import Paths

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
paths = Paths()
classes = {1: "Company", 2: "Educational Institution", 3: "Artist", 4: "Athlete", 5: "Office Holder",
           6: "Mean Of Transportation", 7: "Building", 8: "Natural Place", 9: "Village", 10: "Animal",
           11: "Plant", 12: "Album", 13: "Film", 14: "Written Work"}


class TextCategorizer(object):
    def __init__(self, ARGS):
        # model params
        self.seq_length = ARGS.seq_length
        self.embed_size = ARGS.w2v_embed_size
        self.num_epochs = ARGS.num_epochs
        self.batch_size = ARGS.batch_size

        # load W2V embeddings
        self.vectorizer = WordVectorizer(ARGS)
        self.vectorizer.load_w2v()
        self.stop_words = stopwords.words("english")

        # RNN model
        with tf.Graph().as_default() as self.graph:
            self.model = RNN_Model(len(self.vectorizer.w2v_embeddings.vocab), ARGS)
            self.model.create_placeholders()
            self.model.create_cell()
            self.model.create_loss()
            self.model.create_optimizer()
            self.model.create_predictor()

            # to save the checkpoints
            self.saver = tf.train.Saver(filename=paths.checkpoint + "/" + self.model.name + "-" + str(self.num_epochs))
            # save_relative_paths=True)

    def categorize(self, raw_doc):
        # preprocess and transform
        doc = preprocess_doc(raw_doc[0], self.stop_words)
        doc_input = embedding_lookup([doc], self.vectorizer.w2v_embeddings, self.seq_length, self.embed_size)

        with tf.Session(graph=self.graph) as session:
            # session.run(tf.global_variables_initializer())
            # self.saver = tf.train.import_meta_graph(
            #    paths.checkpoint + "/" + self.model.name + "-" + str(self.num_epochs) + ".meta", clear_devices=True)
            self.saver.restore(session, paths.checkpoint  + self.model.name + "-" + str(self.num_epochs))

            # run the network for each document
            # predicted_labels = list()
            # for doc_input in doc_inputs:
            #    [predicted_label] = session.run([self.model.prediction], {self.model.inputs: doc_input})
            #    predicted_labels.append(predicted_label[0])
            [predicted_label] = session.run([self.model.prediction], {self.model.inputs: doc_input})

            # return the predicted class
            return classes[predicted_label[0]]


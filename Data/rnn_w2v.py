# import libraries
import os
import argparse
import numpy as np
import random
import string
import tensorflow as tf
import pickle
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
import pandas
from word_embedder_gensim import WordVectorizer
from batch_generator import DocumentBatchGenerator
from batch_generator import embedding_lookup
from paths import Paths

# Data paths
paths = Paths()

NUM_CLASSES = 15


class RNN_Model(object):
    """
    Defining RNN Model
    """
    def __init__(self, vocab_size, ARGS):
        
        """
LSTM
Input: documents as sequences of words
Output: class label
"""
        
        # model params
        self.num_hidden = ARGS.hidden_units
        self.num_layers = ARGS.hidden_layers
        self.batch_size = ARGS.batch_size
        self.drop_prob = ARGS.dropout_prob
        self.init_learning_rate = ARGS.learning_rate
        
        self.embed_size = ARGS.w2v_embed_size
        self.vocabulary_size = vocab_size

        # model name as string
        self.name="RNN_hu" + str(ARGS.hidden_units) + "_hl" + str(ARGS.hidden_layers) + "_bs" + str(ARGS.batch_size) + "_dp" + str(ARGS.dropout_prob) + "_lr" + str(ARGS.learning_rate) + "_we" + str(ARGS.w2v_embed_size) + "_ww" + str(ARGS.w2v_window)

    def create_placeholders(self):
        with tf.name_scope("input"):
            self.inputs = tf.placeholder(tf.float32, [None, None, self.embed_size], name="inputs")  
            self.labels = tf.placeholder(tf.int32, [None, NUM_CLASSES], name="labels")

    def _create_single_cell(self):
        # Single cell with dropout
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.num_hidden, state_is_tuple=True),
                                             output_keep_prob=(1-self.drop_prob))
                                             
    def create_cell(self):
        with tf.name_scope("cell"):
            # Create layers
            self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self._create_single_cell() for _ in range(self.num_layers)])

            # State tensor: to pass from sequence to sequences, initialized with zeros
            self.init_state = self.stacked_cell.zero_state(tf.shape(self.inputs)[0], tf.float32)
            
            # Unrolling of LSTM
            output, state = tf.nn.dynamic_rnn(self.stacked_cell, self.inputs,
                                              initial_state=self.init_state, dtype=tf.float32)
            
            # getting latest output
            last_index = tf.shape(output)[1] - 1
            output = tf.transpose(output, [1, 0, 2])
            self.lstm_outputs = tf.gather(output, last_index)
            
            # to store the final state
            self.lstm_state = state
 
    def create_loss(self):
        with tf.name_scope("loss"):
            softmax_W = tf.Variable(tf.truncated_normal([NUM_CLASSES, self.num_hidden]), name="softmax_W")
            softmax_b = tf.Variable(tf.zeros([NUM_CLASSES]), name="softmax_b")

            # Negative subsampling
            self.logits = tf.matmul(self.lstm_outputs, tf.transpose(softmax_W)) + softmax_b
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.labels, self.logits))

    def create_optimizer(self):
        with tf.name_scope("optimizer"):
            # @TODO implement gradient clipping
            self.global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(self.init_learning_rate,
                                                            self.global_step, 2000, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.minimize = optimizer.minimize(self.loss, global_step=self.global_step)

    def create_predictor(self):
        self.prediction = tf.argmax(tf.nn.softmax(self.logits), axis=1)
# end RNN_Model


def run_training(model, embeddings, session, saver, ARGS):
    """ Run minibatch training and save checkpoints """

    # training controlling params
    summary_frequency = int(ARGS.num_epochs / 50) # prints 50 intermediate summaries
    validation_frequency = int(ARGS.num_epochs/ 10)    # validate 10 times
    saver_frequency = int(ARGS.num_epochs/ 5)     # saves 5 checkpoints

    # create training and validation batches
    train_batches = DocumentBatchGenerator(embeddings, ARGS)

    # loading test documents
    print("Loading validation docs...", end="", flush=True)
    if not ARGS.raw_data:
        valid_labels, _, valid_docs = pickle.load(open(paths.data+"/validation_preprocessed.pickle", "rb"))
        valid_labels = np.array(valid_labels)
    else:
        dbvalid = pandas.read_csv(paths.data+"/validation.csv", names=["label", "title","content"])
        valid_docs = list(dbvalid["content"])
        valid_labels = np.array(list(dbvalid["label"]))
    print("done")
    
    # transform validation documents to W2V embeddings
    valid_inputs = embedding_lookup(valid_docs, embeddings, ARGS.seq_length, ARGS.w2v_embed_size)

    mean_loss = 0.0
    state = None # LSTM state
    for step in range(1, ARGS.num_epochs):
        batch_inputs, batch_labels = train_batches.next()
        # converting labels to one-hot
        batch_labels_onehot = np.zeros((batch_labels.size, NUM_CLASSES))
        batch_labels_onehot[np.arange(batch_labels.size), batch_labels] = 1
        # for the first step, initial_state is initialized to zero
        if state == None:
            print("LSTM states are initialized.")
            feed_dict = {model.inputs: batch_inputs, model.labels: batch_labels_onehot}
        else:
            feed_dict = {model.inputs: batch_inputs, model.labels: batch_labels_onehot, model.init_state: state}
            
        _, state, l = session.run([model.minimize, model.lstm_state, model.loss], feed_dict=feed_dict)
        mean_loss += l
        # Printing summaries and savine the model
        if step % summary_frequency == 0:
            # mean loss is an estimate of the loss over the last few batches.
            mean_loss = mean_loss / summary_frequency
            print("STEP " + str(step) + ": average loss = " + str(round(mean_loss, 4)))
            mean_loss = 0

        if step % validation_frequency == 0:
            # measure accuracy on validation data
            [predicted_labels] = session.run([model.prediction], {model.inputs: valid_inputs})
            valid_accuracy = accuracy_score(valid_labels, predicted_labels)
            print("STEP " + str(step) + ": validation accuracy = " + str(round(valid_accuracy*100, 2)) + "%")

        if step % saver_frequency == 0:
            # save checkpoint
            print("STEP " + str(step) + ": saving checkpoint.")
            saver.save(session, paths.checkpoint + "/" + model.name, global_step=model.global_step)

    # Save the final model
    saver.save(session, paths.checkpoint + "/" + model.name, global_step=ARGS.num_epochs)
# end run_training


def run_testing(model, embeddings, session, saver, ARGS):
    """ Run minibatch training and save checkpoints """
    saver.restore(session, paths.checkpoint + "/" + model.name + "-" + str(ARGS.num_epochs))

    # loading test documents
    print("Loading test docs...", end="", flush=True)
    if not ARGS.raw_data:
        test_labels, _, test_docs = pickle.load(open(paths.data+"/test_preprocessed.pickle", "rb"))
        test_labels = np.array(test_labels)
    else:
        dbtest = pandas.read_csv(paths.data+"/test.csv", names=["label", "title","content"])
        test_docs = list(dbtest["content"])
        test_labels = np.array(list(dbtest["label"]))
    print("done")

    # transform documents into W2V embeddings
    test_inputs = embedding_lookup(test_docs, embeddings, ARGS.seq_length, ARGS.w2v_embed_size)

    # run the network
    [predicted_labels] = session.run([model.prediction], {model.inputs: test_inputs})

    # measure and report accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)
    print("Test accuracy = " + str(round(accuracy*100, 2)) + "%")
# end run_testing


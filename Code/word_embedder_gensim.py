"""
Script to generate and store W2V using GENSIM
"""
import os
import logging
import pickle
import argparse
from tqdm import tqdm  # printing progress
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import pandas
from paths import Paths

# Data location
paths = Paths()


class W2VDocumentIterator(object):
    """
    Memory friendly iterator to send a list of words of one document at a time.
    Specific to get used by Word_Vectorizer class
    """

    def __init__(self, ARGS):
        if not ARGS.raw_data:
            _, _, x_train = pickle.load(open(paths.data + "/train_preprocessed.pickle", "rb"))
            _, _, x_valid = pickle.load(open(paths.data + "/validation_preprocessed.pickle", "rb"))
            _, _, x_test = pickle.load(open(paths.data + "/test_preprocessed.pickle", "rb"))
        else:
            dbtrain = pandas.read_csv(paths.data + "/train.csv", names=["label", "title", "content"])
            dbvalid = pandas.read_csv(paths.data + "/validation.csv", names=["label", "title", "content"])
            dbtest = pandas.read_csv(paths.data + "/test.csv", names=["label", "title", "content"])
            x_train = list(dbtrain["content"])
            x_valid = list(dbvalid["content"])
            x_test = list(dbtest["content"])

        self.doc_list = x_train + x_valid + x_test

        # to keep track of progress for this iterator
        self.num_pass = 0

    def __iter__(self):
        progress_string = "PASS " + str(self.num_pass)
        for i_doc in tqdm(range(len(self.doc_list)), desc=progress_string):
            # to keep track of number of passes made on this iterator
            if i_doc == (len(self.doc_list) - 1):
                self.num_pass += 1

            yield self.doc_list[i_doc].split()


# end Document_Iterator


class WordVectorizer(object):
    """
    Creates word2vec vectors for the corpus.
    """

    def __init__(self, ARGS):
        # model params
        self.embedding_size = ARGS.w2v_embed_size  # Dimension of the embedding vector.
        self.window = ARGS.w2v_window  # How many words to consider left and right.

        # model and embedding name as string
        if not ARGS.raw_data:
            self.model_name = "gensimw2vprep_model_emb" + str(self.embedding_size) + "_win" + str(self.window)
            self.embedding_name = "gensimw2vprep_vectors_emb" + str(self.embedding_size) + "_win" + str(self.window)
        else:
            self.model_name = "gensimw2v_model_emb" + str(self.embedding_size) + "_win" + str(self.window)
            self.embedding_name = "gensimw2v_vectors_emb" + str(self.embedding_size) + "_win" + str(self.window)

    def train_w2v(self, ARGS):
        """
        Creates and trains and stores a W2V model.
        """
        # delete if the pickles already exists
        if os.path.isfile(paths.model + "/" + self.model_name) or os.path.isfile(
                paths.model + "/" + self.embedding_name):
            print("Trained files already exist. Deleting... ", end="", flush=True)
            files = os.listdir(paths.model + "/")
            for f in files:
                if f.startswith(self.model_name): os.remove(paths.model + "/" + f)
                if f.startswith(self.embedding_name): os.remove(paths.model + "/" + f)
            print("done")

            # document iterator
        print("Loading documents... ", end="", flush=True)
        documents = W2VDocumentIterator(ARGS)

        print("Training GENSIM W2V model...")
        print("embedding size = " + str(self.embedding_size) + ", skip window = " + str(self.window))
        self.w2v_model = Word2Vec(documents, size=self.embedding_size, window=self.window, min_count=1, workers=4)
        self.w2v_embeddings = self.w2v_model.wv

        # save model and embedding
        self.save_w2v()

    def save_w2v(self):
        """
        Saves the model and the embeddings.
        Note: the model contains the embeddings, still embeddings are saved so that only it can be loaded for querying.
        """
        # save the model
        print("Saving trained W2V model...", end="", flush=True)
        self.w2v_model.save(paths.model + "/" + self.model_name)
        print("done")
        # save the trained embeddings
        print("Saving trained W2V embeddings...", end="", flush=True)
        self.w2v_embeddings.save(paths.model + "/" + self.embedding_name)
        print("done")

    def load_w2v(self, load_model=False):
        """
        Loads a already trained W2V embedding and model (id load_model is True).
        """
        self.w2v_embeddings = KeyedVectors.load(paths.model + self.embedding_name)
        if load_model:
            self.w2v_model = Word2Vec.load(paths.model + self.model_name)
        else:
            self.w2v_model = None



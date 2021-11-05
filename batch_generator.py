"""
Classes to generate batches for training RNN.
1. DocumentBatchGenerator
2. WordBatchGenerator
"""
import argparse
import pickle
import numpy as np
from sklearn.utils import shuffle
from itertools import cycle

from paths import Paths
from word_embedder_gensim import WordVectorizer

# Data location
paths = Paths()


class DocumentBatchGenerator(object):
    """
    Generate the document-wise batches for training RNN.
    1. first get batch_size number of dics
       a) determine the max_length of these docs
       b) pad all docs by 0 vector so that all docs are of the fixed length (here max_length)
    2. labels are documents one shifted 
    """
    def __init__(self, embeddings, ARGS):
        # Gensim trained embeddings
        self.embeddings = embeddings
        self.embed_size = ARGS.w2v_embed_size
        
        # model params
        self.batch_size = ARGS.batch_size
        self.max_doc_length = ARGS.seq_length

        # loading docs
        self.preprocessed = not ARGS.raw_data
        self.load_docs()

    def load_docs(self):
        """
        Load training documents and labels.
        """
        print("Loading training documents...", end="", flush=True)
        if self.preprocessed:
            label_list, _, doc_list = pickle.load(open(paths.data+"/train_preprocessed.pickle", "rb"))
        else:
            dbtrain = pandas.read_csv(paths.data+"/train.csv", names=["label", "title","content"])
            doc_list = list(dbtrain["content"])
            label_list = list(dbtrain["label"])
            
        labels, docs = shuffle(label_list, doc_list)
        self.docs_pool = cycle(docs)
        self.labels_pool = cycle(labels)
        print("done")

    def next(self):
        """
        Generate a single batch from the current cursor position in the data.
        """ 
        # get next batch_size random documents and also determine the max_length
        batch_docs = [next(self.docs_pool) for _ in range(self.batch_size)]
        batch_labels = [next(self.labels_pool) for _ in range(self.batch_size)]
        
        # zero vector for zero-padding
        embedding_zero_vector =  np.zeros(shape=(self.embed_size), dtype=np.float32)
            
        # prepare batch of documents as sequences of W2V embeddings
        batch_inputs = np.zeros(shape=(self.batch_size, self.max_doc_length, self.embed_size), dtype=np.float32)
        # class labels
        batch_labels = np.array(batch_labels)

        for i in range(len(batch_docs)):
            doc_as_vec = np.zeros(shape=(self.max_doc_length, self.embed_size), dtype=np.float32)
            # get the W2V vectors and int ids for the words in the doc
            doc_as_list = batch_docs[i].split()
            for j in range(self.max_doc_length):
                if j < len(doc_as_list):
                    try:
                        doc_as_vec[j] = self.embeddings[doc_as_list[j]]
                    except KeyError:
                        doc_as_vec[j] = embedding_zero_vector
                else:
                    # zero-pad to fill up till max_length
                    doc_as_vec[j] = embedding_zero_vector
                           
            # store the document in the batch
            batch_inputs[i] = doc_as_vec
            
        return batch_inputs, batch_labels
# end of class DocumentBatchGenerator


def embedding_lookup(doc_list, embeddings, max_doc_length, embed_size):
    """
    Given a list of documents, returns a list of W2V embeddings
    """
    # print("Transforming raw docs to W2V embeddings...")
    # zero vector for zero-padding
    embedding_zero_vector = np.zeros(shape=(embed_size), dtype=np.float32)
            
    # prepare batch of documents as sequences of W2V embeddings
    doc_list_vec = np.zeros(shape=(len(doc_list), max_doc_length, embed_size), dtype=np.float32)

    for i in range(len(doc_list)):
        doc_as_vec = np.zeros(shape=(max_doc_length, embed_size), dtype=np.float32)
        # get the W2V vectors and int ids for the words in the doc
        doc_as_list = doc_list[i].split()
        for j in range(max_doc_length):
            if j < len(doc_as_list):
                try:
                    doc_as_vec[j] = embeddings[doc_as_list[j]]
                except KeyError:
                    doc_as_vec[j] = embedding_zero_vector
            else:
                # zero-pad to fill up till max_length
                doc_as_vec[j] = embedding_zero_vector
                           
        # store the document in the batch
        doc_list_vec[i] = doc_as_vec
    
    return doc_list_vec
# end embedding_lookup


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # RNN model hyperparameters
    arg_parser.add_argument("-b", "--batch_size",    type=int,   default=64,   help="size of every batch")
    arg_parser.add_argument("-s", "--seq_length",    type=int,   default=10,   help="sequence length/unrollings")
    #arg_parser.add_argument("-e", "--num_epochs",    type=int,   default=1001, help="number of epochs for training")
    # W2V hyperparameters
    arg_parser.add_argument("-we", "--w2v_embed_size", type=int, default=128, help="embedding dimension for Word2Vec")
    arg_parser.add_argument("-ww", "--w2v_window",     type=int, default=5,   help="skip window size for Word2Vec")
    # Preprocessing
    arg_parser.add_argument("-rw", "--raw_data", default=False, help="Use unpreprocessed raw data", action = "store_true")

    ARGS = arg_parser.parse_args()

    # Load embeddings
    vectorizer = WordVectorizer(ARGS)
    vectorizer.load_w2v()

    batches = DocumentBatchGenerator(vectorizer.w2v_embeddings, ARGS)
    docs, lbl = batches.next()
    print(docs)
    print(lbl)

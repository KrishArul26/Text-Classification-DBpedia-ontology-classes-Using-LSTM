# Preprocess DBPedia Docs:
#  lowercase, punctuation removal, stopword removal

import re
import pandas
import pickle
from nltk.corpus import stopwords
from tqdm import tqdm
from paths import Paths

# importing different paths
paths = Paths()


def preprocess_list(doc_list):
    # stopword list
    stop_words = stopwords.words("english")
    new_doc_list = list()

    for doc in tqdm(doc_list, desc="Preprocessing"):
        doc = preprocess_doc(doc, stop_words)
        new_doc_list.append(doc)

    return new_doc_list
# end preprocess_list


def preprocess_doc(doc, stop_words):
    # 1. remove all numeric references of form [XX]
    doc = re.sub('[\[].[0-9]*[\]]', '', doc)
    doc = re.sub('[\(].*?[\)]', '', doc)
    # 2. remove newlines and multiple whitespace, lower case everything
    doc = re.sub('\s+', ' ', doc).strip()
    doc = doc.lower()

    # 3. remove special characters
    # Regex to keep . , and ' is [^A-Za-z0-9.,\' ]
    doc = re.sub('[^A-Za-z0-9 ]', '', doc)

    # 4. remove stopwords
    doc = " ".join([w for w in doc.split() if w not in stop_words])

    return doc
# end preprocess_doc


def write_csv(labels, titles, contents, mode="train"):
    f = open(paths.data_path + mode + "_preprocessed.csv", "w")

    for i in range(len(labels)):
        f.write(str(labels[i]) + ",\"" + titles[i] + "\",\"" + contents[i] + "\"\n")

    f.close()
# end write_csv


def write_pkl(labels, titles, contents, mode="train"):
    fname = paths.data_path + mode + "_preprocessed.pickle"
    pickle.dump((labels, titles, contents), open(fname, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
# end write_pkl

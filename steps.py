import argparse
import tensorflow as tf
import pandas
from TextCategorizer import TextCategorizer
from paths import Paths
from preprocess import preprocess_list, write_pkl
from rnn_w2v import run_testing, run_training, RNN_Model
from word_embedder_gensim import WordVectorizer

paths = Paths()
if __name__ ==  "__main__":

    # Data Preprocessing starts

    print("\n\n\n" + "=" * 50 + "*"*50 + "=" * 50)
    print("="*50 + "Data Preprocessing starts here" + "="*50)
    # Data Pre Processing
    # Load data
    print("Loading docs...", end="", flush=True)
    dbtrain = pandas.read_csv(paths.data + "train.csv", names=["label", "title", "content"])
    dbvalid = pandas.read_csv(paths.data + "validation.csv", names=["label", "title", "content"])
    dbtest = pandas.read_csv(paths.data + "test.csv", names=["label", "title", "content"])

    train_docs = list(dbtrain["content"])
    valid_docs = list(dbvalid["content"])
    test_docs = list(dbtest["content"])
    print("done")

    # preprocess and create new csv files
    print("Training docs...")
    new_train_docs = preprocess_list(train_docs)
    write_pkl(list(dbtrain["label"]), list(dbtrain["title"]), new_train_docs, mode="train")

    print("Validation docs...")
    new_valid_docs = preprocess_list(valid_docs)
    write_pkl(list(dbvalid["label"]), list(dbvalid["title"]), new_valid_docs, mode="validation")

    print("Test docs...")
    new_test_docs = preprocess_list(test_docs)
    write_pkl(list(dbtest["label"]), list(dbtest["title"]), new_test_docs, mode="test")

    print("=" * 50 + "Data Preprocessing ends here" + "=" * 50)
    print("=" * 50 + "*" * 50 + "=" * 50)

    # Data Preprocessing ends here

    # Model Training Starts here

    print("\n\n\n" + "=" * 50 + "*" * 50 + "=" * 50)
    print("=" * 50 + "Data Training starts here" + "=" * 50)

    arg_parser = argparse.ArgumentParser()

    # W2V hyperparameters
    arg_parser.add_argument("-we", "--w2v_embed_size", type=int, default=128, help="embedding dimension for Word2Vec")
    arg_parser.add_argument("-ww", "--w2v_window", type=int, default=5, help="skip window size for Word2Vec")
    # Preprocessing
    arg_parser.add_argument("-rw", "--raw_data", default=False, help="Use unpreprocessed raw data", action="store_true")

    ARGS = arg_parser.parse_args()

    # train W2V with Gensim
    w2v = WordVectorizer(ARGS)
    w2v.train_w2v(ARGS)

    print("=" * 50 + "Data Training ends here" + "=" * 50)
    print("=" * 50 + "*" * 50 + "=" * 50)
    #  Model Training Ends here

    # RNN LSTM Model Training Starts here


    print("\n\n\n" + "=" * 50 + "*" * 50 + "=" * 50)
    print("=" * 50 + "RNN LSTM Model Training starts here" + "=" * 50)
    arg_parser = argparse.ArgumentParser()
    # RNN model hyperparameters
    arg_parser.add_argument("-b", "--batch_size",    type=int,   default=64,    help="size of every batch")
    arg_parser.add_argument("-s", "--seq_length",    type=int,   default=10,    help="sequence length/unrollings")
    arg_parser.add_argument("-e", "--num_epochs",    type=int,   default=10001, help="number of epochs for training")

    arg_parser.add_argument("-u", "--hidden_units",  type=int,   default=128,   help="number of units in the hidden layers")
    arg_parser.add_argument("-l", "--hidden_layers", type=int,   default=1,     help="number of hidden layers")
    arg_parser.add_argument("-d", "--dropout_prob",  type=float, default=0.5,   help="dropout probability while training")

    arg_parser.add_argument("-r", "--learning_rate", type=float, default=10.0,  help="initial learning rate")

    # W2V hyperparameters
    arg_parser.add_argument("-we", "--w2v_embed_size", type=int, default=128, help="embedding dimension for Word2Vec")
    arg_parser.add_argument("-ww", "--w2v_window",     type=int, default=5,   help="skip window size for Word2Vec")

    # Running parameters
    arg_parser.add_argument("-rw", "--raw_data", help="Use unpreprocessed raw data", action = "store_true")
    arg_parser.add_argument("-te", "--testing", help="flag to run the netwoek in testing mode", action="store_true")

    ARGS = arg_parser.parse_args()

    # Load vocabulry and embeddings
    vectorizer = WordVectorizer(ARGS)
    vectorizer.load_w2v()

    # create rnn graph
    model = RNN_Model(len(vectorizer.w2v_embeddings.vocab), ARGS)
    model.create_placeholders()
    model.create_cell()
    model.create_loss()
    model.create_optimizer()
    model.create_predictor()
    print("Model graph created")

    # to save the checkpoints
    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as session:
        writer = tf.summary.FileWriter("./graph", session.graph)

        session.run(tf.global_variables_initializer())
        print("Variables initialized")

        # Training
        if not ARGS.testing:
            run_training(model, vectorizer.w2v_embeddings, session, saver, ARGS)
        # Testing
        else:
            run_testing(model, vectorizer.w2v_embeddings, session, saver, ARGS)

    print("=" * 50 + "RNN LSTM Model Training ends here" + "=" * 50)
    print("=" * 50 + "*" * 50 + "=" * 50)
    # RNN LSTM Model Training Ends here

    # Model Prediction Starts here

    print("\n\n\n" + "=" * 50 + "Model Prediction starts here" + "=" * 50)
    print("=" * 50 + "*" * 50 + "=" * 50)
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

    text_cat = TextCategorizer(ARGS)

    # get raw doc
    raw_doc = input("\nENTER DOCUMENT: ")
    pred_class = text_cat.categorize([raw_doc])
    print("\n\nPREDICTED CLASS: " + pred_class + "\n")

    print("=" * 50 + "Model Prediction ends here" + "=" * 50)
    print("=" * 50 + "*" * 50 + "=" * 50)

    # RNN LSTM Model Training Ends here



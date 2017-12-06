from src.data_utility import *
import json
import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dropout, MaxPooling1D, Dense, GlobalMaxPool1D, LSTM, Conv1D, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import sys

try:
   import cPickle as pickle
except:
   import pickle


if __name__== '__main__':
    """
    Beginning of experiments with CNN without batches, with pretrained embeddings.
    """
    database_path = 'train/'
    #download_data(database_path)
    embeddings_path = 'embeddings/'
    #download_glove(embeddings_path)
    #unzip_glove("embeddings/", "glove.6B.zip")

    embeddings = get_glove_embeddings(200, embeddings_path)

    #process_data(database_path)
    #build_dictionary(database_path)
    # vectorize_data(database_path)

    word_to_index_pickle_file = "dictionary.pickle"
    if os.path.exists(word_to_index_pickle_file):
        with open(word_to_index_pickle_file, "rb") as f:
            word_to_index = pickle.load(f)
    else:
        word_to_index = json.loads(open("dictionary.json").read())
        with open(word_to_index_pickle_file, "wb") as f:
            pickle.dump(word_to_index, f)

    dict_size = len(word_to_index.keys())
    vectorized_data_path = "train/REUTERS_CORPUS_2/vectorized/"
    tags_path = "train/REUTERS_CORPUS_2/tags/"
    n_train = 3000
    n_test = 3000
    (news_train, tags_train, news_test, tags_test) = get_vectorized_data(vectorized_data_path,
                                                                         tags_path, n_train,
                                                                         n_test,
                                                                         seed=1234)


    lengths = np.array([len(x) for x in news_train])
    max_news_length = int(np.percentile(lengths, 90))
    news_train = sequence.pad_sequences(news_train, maxlen=max_news_length, padding="post")
    news_test = sequence.pad_sequences(news_test, maxlen=max_news_length, padding="post")

    (topics, topic_index, topic_labels) = read_topics(database_path)
    n_class = len(topics)
    # encode responses
    tags_train_matrix = np.zeros((n_train, n_class))
    for ii in range(n_train):
        tags_train_matrix[ii, list(tags_train[ii])] = 1

    tags_test_matrix = np.zeros((n_train, n_class))
    for ii in range(n_test):
        tags_test_matrix[ii, list(tags_test[ii])] = 1


    embedding_matrix = np.zeros((len(word_to_index.keys())+1, 200))
    for word, i in word_to_index.items():
        vector = embeddings.get(word)
        if vector is not None:
            embedding_matrix[i] = vector


    model = Sequential()
    embedding_layer = Embedding(len(word_to_index.keys())+1,
                                200,
                                weights=[embedding_matrix],
                                input_length=max_news_length,
                                trainable=False)

    model.add(embedding_layer)
    model.add(Conv1D(256, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(128, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(np.array(news_train), np.array(tags_train_matrix), epochs=3, batch_size=64)

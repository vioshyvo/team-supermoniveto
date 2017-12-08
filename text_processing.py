from src.data_utility import *
import json
import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dropout, MaxPooling1D, Dense, GlobalMaxPool1D, LSTM, Conv1D, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import sys
from sklearn.metrics import f1_score
from text_generator import *
try:
   import cPickle as pickle
except:
   import pickle


if __name__== '__main__':
    database_path = 'train/'
    embeddings_path = 'embeddings/'
    word_to_index_pickle_file = "dictionary.pickle"
    corpus_path = "train/REUTERS_CORPUS_2/"

    #download_data(database_path)
    #download_glove(embeddings_path)
    #unzip_glove("embeddings/", "glove.6B.zip")
    #process_data(database_path)
    #build_dictionary(database_path)
    #vectorize_data(database_path)

    if os.path.exists(word_to_index_pickle_file):
        with open(word_to_index_pickle_file, "rb") as f:
            word_to_index = pickle.load(f)
    else:
        word_to_index = json.loads(open("dictionary.json").read())
        with open(word_to_index_pickle_file, "wb") as f:
            pickle.dump(word_to_index, f)

    dict_size = len(word_to_index.keys())
    batch_size = 128
    max_news_length = 300
    (topics, topic_index, topic_labels) = read_topics(database_path)
    n_class = len(topics)

    embeddings = get_glove_embeddings(200, embeddings_path)
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

    # model.add(Conv1D(256, 5))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(128, 5))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(n_class, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    # model.fit(np.array(news_train), np.array(tags_train_matrix), epochs=3, batch_size=64)

    model.add(LSTM(100))
    model.add(Dense(n_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    #model.fit(np.array(news_train), np.array(tags_train_matrix), epochs=3, batch_size=64)

    train_files, validation_files, test_files = split_data()
    train_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, train_files)
    validation_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, validation_files)
    train_steps = round(len(train_files) / batch_size)
    validation_steps = round(len(validation_files / batch_size))
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        epochs=3)

    test_seq_matrix, news_tags_matrix = read_file_batch(n_class, max_news_length, corpus_path, test_files)

    prob_test = model.predict(np.array(test_seq_matrix), batch_size=batch_size)
    pred_test = np.array(prob_test) > 0.2
    print('F1 score: ', round(f1_score(news_tags_matrix, pred_test, average='micro'), 4))

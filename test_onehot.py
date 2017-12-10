from src.data_utility import read_topics, split_data
import json
import numpy as np
import random
import os
from keras.layers import Dropout, Dense, Activation
from keras.models import Sequential
from sklearn.metrics import f1_score
from text_generator import text_generator, read_file_batch
try:
   import cPickle as pickle
except:
   import pickle


if __name__== '__main__':
    random.seed(1234)
    database_path = 'train/'
    embeddings_path = 'embeddings/'
    word_to_index_pickle_file = "dictionary.pickle"
    corpus_path = "train/REUTERS_CORPUS_2/"

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

    model = Sequential()
    model.add(Dense(512, input_shape = (dict_size, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
 
    train_files, validation_files, test_files = split_data()
    print(len(train_files))
    train_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, train_files)
    train_steps = round(len(train_files) / batch_size)

    print("Train steps", train_steps)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps,
                        epochs=2)

    test_seq_matrix, news_tags_matrix = read_file_batch(n_class, max_news_length, corpus_path, test_files)

    prob_test = model.predict(np.array(test_seq_matrix), batch_size=batch_size)
    pred_test = np.array(prob_test) > 0.2
    print('F1 score: ', f1_score(news_tags_matrix, pred_test, average='micro'))


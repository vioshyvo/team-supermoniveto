from src.data_utility import *
import json
import numpy as np
import keras
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
from matplotlib import pyplot as plt
from IPython.display import clear_output


# Tensorflow memory consumption limit. Uncomment if needed.
# import tensorflow as tf
# from keras import backend as k
# config = tf.ConfigProto()
# # Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True
# # Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# # Create a session with the above options specified.
# k.tensorflow_backend.set_session(tf.Session(config=config))


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        #
        # clear_output(wait=True)
        # plt.plot(self.x, self.losses, label="loss")
        # plt.plot(self.x, self.val_losses, label="val_loss")
        # plt.legend()
        # plt.show()

    def on_train_end(self, logs=None):
        clear_output(wait=True)
        plt.subplot(211)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()

        plt.subplot(212)
        plt.plot(self.x, self.acc, label="accuracy")
        plt.plot(self.x, self.val_acc, label="val_accuracy")
        plt.legend()
        plt.show()

plot_losses = PlotLosses()


if __name__== '__main__':
    random.seed(1234)
    database_path = 'train/'
    test_database_path = 'test/'
    embeddings_path = 'embeddings/'
    word_to_index_pickle_file = "dictionary.pickle"
    corpus_path = "train/REUTERS_CORPUS_2/"

    #download_data(database_path)
    #download_glove(embeddings_path)
    #unzip_glove("embeddings/", "glove.6B.zip")
    #process_data(database_path)
    #build_dictionary(database_path)
    #vectorize_data(database_path)
    #coalesce_data(database_path)

    if os.path.exists(word_to_index_pickle_file):
        with open(word_to_index_pickle_file, "rb") as f:
            word_to_index = pickle.load(f)
    else:
        word_to_index = json.loads(open("dictionary.json").read())
        with open(word_to_index_pickle_file, "wb") as f:
            pickle.dump(word_to_index, f)

    dict_size = len(word_to_index.keys())
    batch_size = 256
    max_news_length = 300
    (topics, topic_index, topic_labels) = read_topics(database_path)
    
    index2topic = dict()
    for k, v in topic_index.items():
        index2topic[v] = k
     
    n_class = len(topics)

    embedding_size = 300

    embeddings = get_glove_embeddings(embedding_size, embeddings_path)
    embedding_matrix = np.zeros((len(word_to_index.keys())+1, embedding_size))
    for word, i in word_to_index.items():
        vector = embeddings.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
        else:
            embedding_matrix[i] = np.random.normal(loc=0.0, scale=1.0, size=embedding_size)

    model = Sequential()
    embedding_layer = Embedding(len(word_to_index.keys())+1,
                                embedding_size,
                                weights=[embedding_matrix],
                                input_length=max_news_length,
                                trainable=False)

    model.add(embedding_layer)

    #1
    # model.add(Conv1D(100, 4))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(100, 2))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(n_class, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 3 epochs
    # F1 score:  0.829635600654

    #2
    # model.add(Conv1D(300, 4, activation='relu'))
    # model.add(Conv1D(100, 6, activation='relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(100, 10, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_class, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 3 epochs, batch size=64:
    # F1 score:  0.830657251065
    # 5 epochs, batch size=64:
    # F1 score:  0.830636177074

    # 3 epochs, batch size=128:
    # F1 score:  0.836862952723
    # 5 epochs, batch size=128:
    # F1 score:  0.842977576702

    # 3 epochs, batch size=256:
    # F1 score:  0.841250447014
    # 5 epochs, batch size=256:
    # F1 score:  0.846580333452

    # 3 epochs, batch size=256, Glove=300:
    # F1 score:  0.844318673503

    # 5 epochs, batch size=256, Glove=300, unknown words handling:
    # F1 score:  0.84929320338

    #3
    model.add(Conv1D(300, 4, activation='relu'))
    model.add(Conv1D(100, 6, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 
    # 5 epochs, batch size=256, Glove=300, unknown words handling:
    # F1 score:  0.849875387055

    # 5 epochs, batch size=256, Glove=300, unknown words handling, threshold=0.4:
    # F1 score:  0.853655083795

    # 5 epochs, batch size=256, Glove=300, unknown words handling, threshold=0.5:
    # F1 score:  0.855645045463 +++

    # 5 epochs, batch size=256, Glove=300, unknown words handling, threshold=0.6:
    # F1 score:  0.846565697727

    #4
    # model.add(Conv1D(300, 4, activation='relu'))
    # model.add(Conv1D(100, 6, activation='relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(100, 10, activation='relu'))
    # model.add(Flatten())
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_class, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 5 epochs:
    # F1 score:  0.830928370233

    #5
    # model.add(Conv1D(300, 8, activation='relu'))
    # model.add(Conv1D(100, 16, activation='relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(50, 32, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_class, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # F1 score:  0.816741292086

    #6
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(64))
    #model.add(Dense(n_class, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    #model.fit(np.array(news_train), np.array(tags_train_matrix), epochs=3, batch_size=64)

    with open(corpus_path + 'coalesced_data.pickle', 'rb') as f:
        data_cache = pickle.load(f)

    train_files, validation_files, test_files = split_data()
    print(len(train_files))
    train_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, train_files, data_cache)
    validation_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, validation_files, data_cache)
    train_steps = round(len(train_files) / batch_size)
    validation_steps = round(len(validation_files) / batch_size)

    print("Train steps", train_steps)
    print("Validation steps", validation_steps)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        epochs=5,
                        callbacks=[plot_losses]
                        )

    test_seq_matrix, news_tags_matrix = read_file_batch(n_class, max_news_length, corpus_path, test_files, data_cache)

    prob_test = model.predict(np.array(test_seq_matrix), batch_size=batch_size)
    pred_test = np.array(prob_test) > 0.5
    print('F1 score: ', f1_score(news_tags_matrix, pred_test, average='micro'))
    model.save("best_model.h5") 
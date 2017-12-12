#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

from src.data_utility import read_topics, get_glove_embeddings, split_data, download_test
import json
import numpy as np
import os
import random
from keras.layers import Embedding, Dropout, MaxPooling1D, Dense, Conv1D, Activation, Flatten, Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.metrics import f1_score
from text_generator import text_generator, read_file_batch
try:
   import cPickle as pickle
except:
   import pickle


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


random.seed(1234)
database_path = 'train/'
embeddings_path = 'embeddings/'
word_to_index_pickle_file = "dictionary.pickle"
corpus_path = "train/REUTERS_CORPUS_2/"
test_path = 'test/'

#    download_data(database_path)
#    download_glove(embeddings_path)
#    unzip_glove("embeddings/", "glove.6B.zip")
#    process_data(database_path)
#    build_dictionary(database_path)
#    vectorize_data(database_path)
#    coalesce_data(database_path)

if os.path.exists(word_to_index_pickle_file):
    with open(word_to_index_pickle_file, "rb") as f:
        word_to_index = pickle.load(f)
else:
    word_to_index = json.loads(open("dictionary.json").read())
    with open(word_to_index_pickle_file, "wb") as f:
        pickle.dump(word_to_index, f)


filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50


dict_size = len(word_to_index.keys())
batch_size = 256
max_news_length = 300
(topics, topic_index, topic_labels) = read_topics(database_path)
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

model_input = Input(shape=(max_news_length, ))
x = Embedding(len(word_to_index.keys())+1,
                            embedding_size,
                            weights=[embedding_matrix],
                            input_length=max_news_length,
                            trainable=False)(model_input)


# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(x)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

x = Dropout(dropout_prob[1])(x)
x = Dense(hidden_dims, activation="relu")(x)
model_output = Dense(n_class, activation="sigmoid")(x)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

with open(corpus_path + 'coalesced_data.pickle', 'rb') as f:
    data_cache = pickle.load(f)

train_files, validation_files, test_files = split_data()
print(len(train_files))
train_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, train_files, data_cache)
train_steps = round(len(train_files) / batch_size)

print("Train steps", train_steps)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_steps,
                    epochs=5
                    )

test_seq_matrix, news_tags_matrix = read_file_batch(n_class, max_news_length, corpus_path, test_files, data_cache)

prob_test = model.predict(np.array(test_seq_matrix), batch_size=batch_size)
pred_test = np.array(prob_test) > 0.5
print('F1 score: ', f1_score(news_tags_matrix, pred_test, average='micro'))

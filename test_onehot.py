#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

from src.data_utility import read_topics, split_data
import json
import numpy as np
import os
import random
from keras.layers import Dropout, Dense, Activation
from keras.models import Sequential
from sklearn.metrics import f1_score
from text_generator import text_generator, read_tag_batch
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

dict_size = len(word_to_index.keys()) + 1
batch_size = 64
max_news_length = 300
(topics, topic_index, topic_labels) = read_topics(database_path)
n_class = len(topics)


model = Sequential()
model.add(Dense(64, input_shape=(dict_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

with open(corpus_path + 'coalesced_data.pickle', 'rb') as f:
    data_cache = pickle.load(f)

train_files, validation_files, test_files = split_data()
train_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, train_files, data_cache, True, dict_size)
train_steps = round(len(train_files) / batch_size)
test_steps = round(len(test_files) / batch_size)

print("Train steps: ", train_steps, ', Test steps: ', test_steps)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_steps,
                    epochs=10
                    )

#prob_test = model.predict(np.array(test_seq_matrix), batch_size=batch_size)
test_generator = text_generator(batch_size, n_class, max_news_length, corpus_path, test_files, data_cache, True, dict_size)
news_tags_matrix = read_tag_batch(n_class, corpus_path, test_files, data_cache)

<<<<<<< HEAD
prob_test = model.predict_generator(test_generator, test_steps)
thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
for thres in thresholds:
    pred_test = np.array(prob_test) > thres
    print('cutoff: ', thres, ' F1 score: ', f1_score(news_tags_matrix[0:29952], pred_test, average='micro'))

||||||| merged common ancestors
prob_test = model.predict(np.array(test_seq_matrix), batch_size=batch_size)
pred_test = np.array(prob_test) > 0.5
print('F1 score: ', f1_score(news_tags_matrix, pred_test, average='micro'))

=======
prob_test = model.predict(np.array(test_seq_matrix), batch_size=batch_size)
pred_test = np.array(prob_test) > 0.5
print('F1 score: ', f1_score(news_tags_matrix, pred_test, average='micro'))
>>>>>>> 4ae3aa20647b4cf500168fbcb97fa372a6586165

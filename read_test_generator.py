from src.data_utility import download_test, process_data, vectorize_data, read_topics
from text_generator import text_generator_test
from keras.models import load_model
import os
import json
import pickle
import numpy as np

test_path = 'test/'
max_news_length = 300

#download_test(test_path)
#process_data(test_path, False)
#vectorize_data(test_path)

word_to_index_pickle_file = "dictionary.pickle"
database_path = 'train/'

if os.path.exists(word_to_index_pickle_file):
    with open(word_to_index_pickle_file, "rb") as f:
        word_to_index = pickle.load(f)
else:
    word_to_index = json.loads(open("dictionary.json").read())
    with open(word_to_index_pickle_file, "wb") as f:
        pickle.dump(word_to_index, f)

dict_size = len(word_to_index.keys()) + 1
batch_size = 64
(topics, topic_index, topic_labels) = read_topics(database_path)
n_class = len(topics)


##------------------load model and predict -----------------------------##

model = load_model('bow_model.h5')

test_files = os.listdir(test_path + 'REUTERS_CORPUS_2/vectorized/')
test_files.sort()

test_steps = round(len(test_files) / batch_size) + 1
test_generator = text_generator_test(batch_size, max_news_length, test_path, test_files, True, dict_size)
prob_test = model.predict_generator(test_generator, test_steps)

thres = 0.3
pred_test = np.array(prob_test) > thres
 
# rows of the output matrix correspond to the alphabetical order of the test files
np.savetxt('results_bow.txt', pred_test, fmt='%d')


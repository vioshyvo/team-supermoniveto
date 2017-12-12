from src.data_utility import download_test, process_data, vectorize_data, load_vectorized_test, read_test_batch, read_topics
import numpy as np

test_path = 'test/'

download_test(test_path)
process_data(test_path, False)
vectorize_data(test_path)

### ------------------------------- ###
# input trained model here
### ------------------------------- ###

max_news_length = 300
test_seq_matrix = read_test_batch(max_news_length, test_path)

prob_test = model.predict(np.array(test_seq_matrix), batch_size=256)
pred_test = np.array(prob_test) > 0.5


import os 
import shutil

n_test = 10000
test_path2 = 'test2/'
input_path = 'train/REUTERS_CORPUS_2/data/'
output_path = test_path2 + 'REUTERS_CORPUS_2/data/'

file_list = os.listdir(input_path)[0:n_test]
os.makedirs(output_path) 
for file_name in file_list:
    shutil.copy(input_path + file_name, output_path)

file_list2 = os.listdir(output_path)[0:n_test] 
file_list[0:10]
file_list2[0:10]

(topics, topic_index, topic_labels) = read_topics(database_path)
n_class = len(topics)
corpus_path = "train/REUTERS_CORPUS_2/"

file_list3 = ['_' + f for f in file_list]

t, news_tags_matrix = read_file_batch(n_class, max_news_length, corpus_path, file_list3)



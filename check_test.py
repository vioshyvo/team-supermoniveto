import numpy as np
from src.data_utility import read_topics, read_xml_file
import os

"""
Read the results file, and print the predicted tags along with the contents
of the files for the sanity check
"""


results_file = 'results.txt'

database_path = 'train/'
pred = np.loadtxt(results_file, dtype=bool)
(topics, topic_index, topic_labels) = read_topics(database_path)
n_class = len(topics)
n_pred = pred.shape[0]

pred_topics = []
for i in range(n_pred): 
    p = pred[i, ]
    t = [topics[i] for i in range(n_class) if p[i]]
    labels = [topic_labels[topic] for topic in t]
    pred_topics.append(labels)


test_dir = 'test/REUTERS_CORPUS_2/data/'
testfiles = os.listdir(test_dir)
testfiles.sort()
testfiles = [f for f in testfiles if f.endswith('xml')]

# print first 10 test items with predicted labels
for i in range(0,10):
    print(i + 1, ':th file, predicted topics: ', pred_topics[i], '\n')
    msg = read_xml_file(test_dir + testfiles[i])
    print('message:')
    for line in msg[0]:
        print(line)
    print('\n\n---------------------------------------------------\n\n')
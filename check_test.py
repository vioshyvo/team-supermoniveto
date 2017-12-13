import numpy as np
from src.data_utility import read_topics, read_xml_file
import os

database_path = 'train/'
pred = np.loadtxt('results.txt', dtype=bool)
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

f = testfiles[2]
r = read_xml_file(test_dir + f)
rr = r[0]

# print first 10 test items with predicted labels
for i in range(0,10):
    print('predicted topics: ', pred_topics[i], '\n')
    msg = read_xml_file(test_dir + testfiles[i])
    print('message:')
    for line in msg[0]:
        print(line)
    print('\n\n---------------------------------------------------\n\n')
import os
from src.data_utility import *
import numpy as np
import matplotlib.pyplot as plt


"""
Code to make a figure with dictribution of all topics over all documents
"""

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 5,
                '%d' % int(height),
                ha='center', va='bottom',
                rotation="vertical", fontsize=6)

if __name__== '__main__':
    database_path = 'train/'
    corpus_path = "train/REUTERS_CORPUS_2/"
    tags_path = corpus_path + 'tags/'

    data_list = os.listdir(tags_path)[:1000]
    tags, tag_index, __ = read_topics(database_path)
    tag_dict = dict()
    for tag in tags:
        tag_dict[tag] = 0

    for tag_file in data_list:
        file_tags = np.load(tags_path + tag_file)
        for t in file_tags:
            tag_dict[tags[t]] += 1

    tag_list = []
    tag_names = []
    for k, v in sorted(tag_dict.items(), key=lambda x: x[1]):
        #if v <= 4:
        #    continue
        tag_list.append(v)
        tag_names.append(k)
    ind = np.arange(len(tag_list))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, tag_list, width, color='b')
    ax.set_ylabel('Number of texts')
    ax.set_title('Distribution over topics')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tag_names, rotation='vertical', fontsize=6)

    ax.legend((rects1[0],))
    autolabel(rects1)
    plt.show()

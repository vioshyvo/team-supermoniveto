from keras.utils.data_utils import get_file
import os
import zipfile
import random
import xml.etree.ElementTree as etree

    
def download_data(database_path = 'train/'):
    """
    Downloads the data set if it is not yet downloaded (the download path given as argument)
    does not yet exist into the path given as an argument, and unzips the data set
    if it is not yet unzipped.
    """
    corpus_path = database_path + 'REUTERS_CORPUS_2/'
    data_path = corpus_path + 'data/'
    codes_path = corpus_path + 'codes/'
    
    if not os.path.exists(database_path):
        dl_file='reuters.zip'
        dl_url='https://www.cs.helsinki.fi/u/jgpyykko/'
        get_file(dl_file, dl_url+dl_file, cache_dir='./', cache_subdir=database_path, extract=True)
    else:
        print('Data set already downloaded.')

    if not os.path.exists(data_path):
        print('\n\nUnzipping data...')
    
        codes_zip = corpus_path + 'codes.zip'
        with zipfile.ZipFile(codes_zip, 'r') as zip_ref:
            zip_ref.extractall(codes_path)
        os.remove(codes_zip)
   
        dtds_zip = corpus_path + 'dtds.zip'
        with zipfile.ZipFile(dtds_zip, 'r') as zip_ref:
            zip_ref.extractall(corpus_path + 'dtds/')
        os.remove(dtds_zip)
    
        for item in os.listdir(corpus_path): 
            if item.endswith('zip'):
                file_name = corpus_path + item 
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
                os.remove(file_name) 
    
        print('Data set unzipped.')
    else:
        print('Data set already unzipped.')
    

def read_topics(database_path):
    """
    Read topics of the data set into the look - up table. Returns:
    topics = topic codes as an array
    topic_index = dictionary with topic code as key, and index of topic in topics as a value
    topic_labels = dictionary with topic code as key, and topic labels as a value
    """
    topics = []
    topic_labels = {}
    corpus_path = database_path + 'REUTERS_CORPUS_2/'
    codes_path = corpus_path + 'codes/'
    codes_file = codes_path + 'topic_codes.txt'
    with open(codes_file) as f:
        for line in f:
            if not line.startswith(';'):
                splits = line.split()
                topic_code = splits[0]
                topic_labels[topic_code] = ' '.join(splits[1:len(splits)])
                topics.append(topic_code)
    
    n_class = len(topics)
    topic_index = {topics[i] : i for i in range(n_class)}
    
    return (topics, topic_index, topic_labels)

def read_xml_file(file_xml):
    sentences = []
    tags = []
    read_tags = False
    for event, elem in etree.iterparse(file_xml, events=('start', 'end')):
        t = elem.tag
        idx = k = t.rfind("}")
        if idx != -1:
            t = t[idx + 1:]
        tname = t

        if event == 'start':
            if tname == 'codes':
                if elem.attrib['class'] == 'bip:topics:1.0':
                    read_tags = True
            if tname == 'code':
                if read_tags:
                    tags.append(elem.attrib['code'])
    
        if event == 'end':
            if tname == 'headline':
                sentences.append(elem.text)
            if tname == 'p':
                sentences.append(elem.text)
            if tname == 'codes':
                if elem.attrib['class'] == 'bip:topics:1.0':
                    read_tags = False

    return [sentences, tags]
    

def read_news(database_path, n_train, n_test, seed = None):
    """
    Read a file into the training set of size n_train and a test set of size n_test. Returns following lists:
    news_train = news items of training set as lists of sentences
    tags_train = topics of news items of training set as lists
    news_test = news items of test set as lists of sentences
    tags_test = topics of news items of training set as lists
    """
    corpus_path = database_path + 'REUTERS_CORPUS_2/'
    data_path = corpus_path + 'data/'
    
    if seed is not None:
        random.seed(seed)

    data_list = os.listdir(data_path)
    n_samples = len(data_list)
    random_indices = random.sample(range(n_samples), n_train + n_test)

    train_indices = random_indices[0:n_train]
    test_indices = random_indices[n_train:(n_train + n_test)]

    train_list = [data_list[i] for i in train_indices]
    test_list = [data_list[i] for i in test_indices]

    news_train = []
    tags_train = []
    for file_name in train_list:
        file_xml = data_path + file_name 
        (sentences, tags) = read_xml_file(file_xml)
        news_train.append(sentences)
        tags_train.append(tags)
    
    news_test = []
    tags_test = []
    for file_name in test_list:
        file_xml = data_path + file_name 
        (sentences, tags) = read_xml_file(file_xml)
        news_test.append(sentences)
        tags_test.append(tags)

    return (news_train, tags_train, news_test, tags_test)
    
    







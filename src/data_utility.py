from keras.utils.data_utils import get_file
import os
import zipfile
import random
import xml.etree.ElementTree as etree
import numpy as np
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import codecs
import json

try:
    import cPickle as pickle
except:
    import pickle


def download_data(database_path='train/'):
    """
    Downloads the data set if it is not yet downloaded (the download path given as argument)
    does not yet exist into the path given as an argument, and unzips the data set
    if it is not yet unzipped.
    """
    corpus_path = database_path + 'REUTERS_CORPUS_2/'
    data_path = corpus_path + 'data/'
    codes_path = corpus_path + 'codes/'

    if not os.path.exists(database_path):
        dl_file = 'reuters.zip'
        dl_url = 'https://www.cs.helsinki.fi/u/jgpyykko/'
        get_file(dl_file, dl_url + dl_file, cache_dir='./', cache_subdir=database_path, extract=True)
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
    topic_index = {topics[i]: i for i in range(n_class)}
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


def read_news(database_path, n_train, n_test, seed=None):
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


def filter_text(sentences):
    stop_words = set(stopwords.words('english'))
    result = []
    for sentence in sentences:
        if sentence is not None:
            translator = str.maketrans('', '', string.punctuation)
            sentence = sentence.translate(translator)
            word_tokens = word_tokenize(sentence)
            filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]
            for word in filtered_sentence:
                if word.isdigit():
                    result.append('NUM')
                else:
                    result.append(word)
    return result


def process_data(database_path):
    """
    File which saves all files filtered and tokenized and also saves a
    file with np.array including all tags for a file. This is done for a purpose not to re-analyze all data again.
    :param database_path:
    :return: write all docs filtered and tokenised into a file, splitted by a whitespace; write a .npy file with topics
    """
    corpus_path = database_path + 'REUTERS_CORPUS_2/'
    data_path = corpus_path + 'data/'

    tags_path = corpus_path + 'tags/'
    tokenized_path = corpus_path + 'tokenized/'

    tagged = False
    tokenized = False
    if not os.path.exists(tags_path):
        os.makedirs(tags_path)
    else:
        print("Tags are already written")
        tagged = True
    if not os.path.exists(tokenized_path):
        os.makedirs(tokenized_path)
    else:
        print("Tokens are already written")
        tokenized = True
    data_list = os.listdir(data_path)

    (topics, topic_index, topic_labels) = read_topics(database_path)

    if not (tagged and tokenized):
        for file_name in data_list:
            file_xml = data_path + file_name
            (sentences, tags) = read_xml_file(file_xml)

            filtered_sentences = filter_text(sentences)

            tokenized_filename = tokenized_path + '_' + os.path.splitext(file_name)[0] + '.txt'
            tag_filename = tags_path + '_' + os.path.splitext(file_name)[0] + '.npy'

            tag_list = []
            for tag in tags:
                tag_index = topic_index[tag]
                tag_list.append(tag_index)
            tags_array = np.array(tag_list)

            with codecs.open(tokenized_filename, 'w', encoding="utf-8") as tkf:
                tkf.write(' '.join(filtered_sentences))
            np.save(tag_filename, tags_array)


def build_dictionary(database_path):
    """
    A function to build a dictionary of all words in the data with words as keys and their indexes as values.
    0 index is for all new unknown words.
    """
    data_path = database_path + 'REUTERS_CORPUS_2/tokenized/'
    dictionary = dict()
    i = 1
    data_list = os.listdir(data_path)
    for file_name in data_list:
        with codecs.open(data_path + file_name, 'r', encoding="utf-8") as f:
            sentence = f.read()
            tokens = sentence.split(' ')
            for word in tokens:
                if word not in dictionary:
                    dictionary[word] = i
                    i += 1
    with open('dictionary.json', 'w') as df:
        json.dump(dictionary, df)


def vectorize_data(database_path):
    """
    A function to modify a tokenized file into a numpy array of its word indexes.
    """
    data_path = database_path + 'REUTERS_CORPUS_2/tokenized/'
    vectorized_data_path = database_path + 'REUTERS_CORPUS_2/vectorized/'
    if not os.path.exists(vectorized_data_path):
        os.makedirs(vectorized_data_path)

    data_list = os.listdir(data_path)
    with open('dictionary.json') as json_data:
        dictionary = json.load(json_data)
    for file_name in data_list:
        with codecs.open(data_path + file_name, 'r', encoding="utf-8") as f:
            sentence = f.read()
            tokens = sentence.split(' ')
            index_list = []
            for word in tokens:
                index = dictionary.get(word, 0)
                index_list.append(index)
            vector = np.array(index_list)
            np_filename = vectorized_data_path + os.path.splitext(file_name)[0] + '.npy'
            np.save(np_filename, vector)


def get_vectorized_data(vectorized_data_path="train/REUTERS_CORPUS_2/vectorized/",
                        tags_path="train/REUTERS_CORPUS_2/tags/",
                        n_train=3000, n_test=3000, seed=None):
    """
    For getting a sample for training from vectorized data in order to start training
    """
    if seed is not None:
        random.seed(seed)

    data_list = os.listdir(tags_path)
    n_samples = len(data_list)
    random_indices = random.sample(range(n_samples), n_train + n_test)

    train_indices = random_indices[0:n_train]
    test_indices = random_indices[n_train:(n_train + n_test)]

    train_list = [data_list[i] for i in train_indices]
    test_list = [data_list[i] for i in test_indices]

    news_train = []
    tags_train = []
    for file_name in train_list:
        sentences = np.load(vectorized_data_path + file_name)
        tags = np.load(tags_path + file_name)
        news_train.append(sentences)
        tags_train.append(tags)

    news_test = []
    tags_test = []
    for file_name in test_list:
        sentences = np.load(vectorized_data_path + file_name)
        tags = np.load(tags_path + file_name)
        news_test.append(sentences)
        tags_test.append(tags)

    return (news_train, tags_train, news_test, tags_test)


def split_data(data_path="train/REUTERS_CORPUS_2/vectorized/",
               test_proportion=0.1, validation_proportion=0.1, seed=None):
    """
    Function to split data into training, validation and test set. Returns three lists of files
    """
    if seed is not None:
        random.seed(seed)

    data_list = os.listdir(data_path[0:2000])
    n_samples = len(data_list)
    n_test = int(n_samples * test_proportion)
    n_validation = int(n_samples * validation_proportion)
    n_train = n_samples - (n_test + n_validation)

    random_indices = random.sample(range(n_samples), n_samples)
    train_indices = random_indices[0:n_train]
    validation_indices = random.sample[n_train:n_validation]
    test_indices = random_indices[n_validation:(n_samples)]

    train_list = [data_list[i] for i in train_indices]
    validation_list = [data_list[i] for i in validation_indices]
    test_list = [data_list[i] for i in test_indices]

    return train_list, validation_list, test_list


def download_glove(embeddings_path='embeddings/'):
    """
    Download the glove-embeddings, pretty slow so I would recommend to download it on its own and locate it to ./embeddings/ folder
    """
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)
    file_name = "glove.6B.zip"
    zip_file_path = embeddings_path + file_name
    if not os.path.exists(embeddings_path + file_name):
        dl_url = "http://nlp.stanford.edu/data/glove.6B.zip" + file_name
        print("Downloading GloVe embeddings 866MB")
        curl_command = "curl -o " + zip_file_path + " \"https://nlp.stanford.edu/data/glove.6B.zip\""
        os.system(curl_command)
    else:
        print("GloVe Zip found")


def unzip_glove(embeddings_path="embeddings/", zip_file_name="glove.6B.zip"):
    unzipped_names = ['glove.6B.100d.txt', 'glove.6B.200d.txt', 'glove.6B.300d.txt', 'glove.6B.50d.txt', 'glove.6B.zip']
    if not (all([os.path.exists(embeddings_path + i) for i in unzipped_names])):
        print("Unzipping")
        zip_file_path = embeddings_path + zip_file_name
        zip_ref = zipfile.ZipFile(zip_file_path, 'r')
        zip_ref.extractall(embeddings_path)
        zip_ref.close()
    else:
        print("Already unzipped")


def get_glove_embeddings(dimension=200, embeddings_path="embeddings/"):
    """
    Create a dictionary which has the form of embedding["word"] = np.array
    Note that dimension needs to be one of [50,100,200,300]
    """

    pickle_file = "glove_embeddings_" + str(dimension) + ".pickle"
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            embeddings = pickle.load(f)
            return embeddings

    files = os.listdir(embeddings_path)
    file_name = next(file for file in files if file.endswith(str(dimension) + "d.txt"))
    embeddings = {}
    with codecs.open(embeddings_path + file_name, encoding="utf-8") as f:
        for line in f:
            splits = line.split(" ")
            embeddings[splits[0]] = np.array([float(i) for i in splits[1:]])

    with open(pickle_file, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

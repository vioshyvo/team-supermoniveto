import os
import numpy as np
from keras.preprocessing import sequence


def iter_window(iterable, size):
    it = iter(iterable)
    while True:
        result = []
        try:
            for _ in range(size):
                n = next(it)
                result.append(n)
            yield result
        except StopIteration:
            if len(result) > 0:
                yield result
            break


def text_generator(batch_size, n_class, max_text_length, corpus_path, files_to_use):
    """
    A generator for Keras fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    We will use fit_generator instead of fit to work with batches of data.
    We need to do a split of data into train and test first and then to give splitted data to generator,
    generator will return input matrixes for text data and tags of the required batch size.
    :param batch_size:
    :param n_class:
    :param max_text_length:
    :param corpus_path:
    :param files_to_use:
    :return:
    """
    data_path = corpus_path + 'vectorized/'
    tag_path = corpus_path + 'tags/'
    while True:
        for file_batch in iter_window(files_to_use, batch_size):
            data_rows = []
            tag_rows = []
            for f in file_batch:
                data = np.load(data_path + f)
                data = sequence.pad_sequences([data], maxlen=max_text_length, padding='post')
                data_rows.append(data)

                tags = np.load(tag_path + f)
                tag_rows.append(tags)

            n_rows = len(tag_rows)
            data_matrix = np.vstack(data_rows)
            tags_matrix = np.zeros((n_rows, n_class))
            for ii in range(n_rows):
                tags_matrix[ii, list(tag_rows[ii])] = 1

            yield (data_matrix, tags_matrix)
    pass


if __name__== '__main__':
    corpus_path = "train/REUTERS_CORPUS_2/"
    files_to_use = os.listdir(corpus_path + "vectorized/")
    g = text_generator(5, 126, 300, corpus_path, files_to_use)
    print(next(g))
    pass

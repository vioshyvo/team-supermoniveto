from keras.utils.data_utils import get_file
import os
import zipfile

def times_two(x):
    return 2 * x

def download_data(database_path = 'train/'):
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
    



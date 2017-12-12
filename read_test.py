from src.data_utility import download_test, process_data, vectorize_data

test_path = 'test/'
download_test(test_path)
process_data(test_path, False)
vectorize_data(test_path)

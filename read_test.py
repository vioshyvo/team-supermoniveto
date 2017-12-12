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

# rows of the output matrix correspond to the alphabetical order of the test files
np.savetxt('results.txt', pred_test, fmt='%d')


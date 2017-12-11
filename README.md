# Deep learning course project

This repo contains a deep learning course final project of team **supermoniveto**:
- Ville Tanskanen
- Ville Hyvönen
- Anisia Katinskaia

![](veikkaus.JPG)

### Summary

The data we chose for our group was the [text data](https://keras.io/datasets/#reuters-newswire-topics-classification). In the fashion of the course we used deep learning to learn the problem and do predictions. We tried vast range of different models ranging from bag of words multilayer perceptron (MLP) to a combination of different advanced techniques such as CNN and LSTM.

As the data preprocessing step is very important we gave pretty careful attention to it.

Anisia...

For the word embeddings we used the pre-trained [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/). Most of the modelling is done with 200 dimensional representations and the competition model training was done with 300 dimensional representations.

Most of the model verification was done with 20K random sample from the data. We used a 10K/10K split in order for the test set to generalize as well as possible and still keep the training times reasonable for even CPU. First trial was a quick MLP after preprocessing step to ensure that we have one working model. Then after preprocessing we started to experiment with the more advanced techniques involving an embedding layer to our network.
First we tried the last topic of the course, LSTM and its combinations. However during the testing we quickly realized that the traditional CNN was working much better. As it seemed that the CNN was the way to go we experimented a lot of different sets of hyperparameters by hand. After exhaustive search for the best hyper parameters we had some idea of what would be our chosen model for the competition. Being curious, we also expanded our views outside of the course and did some trials with combination of CNN and LSTM where first comes the convolutions and then the LSTM is applied for the convolved layers. Final two models that we tried were bidirectional LSTM, which is used for the sequence classification when the whole sequence is known, and gated recurrent unit (GRU) which should be similar to LSTM but have faster training as it is missing some of the gates from LSTM.

### Future work

TODO

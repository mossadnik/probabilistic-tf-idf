[![Build Status](https://travis-ci.org/mossadnik/probabilistic-tf-idf.svg?branch=master)](https://travis-ci.org/mossadnik/probabilistic-tf-idf)

# probabilistic-tf-idf

In name matching / entity resolution, token-based tf-idf scores are a performant and surprisingly effective tool But they can hardly benefit from available data with validated matches, so that all improvements and domain adaptation has to be done manually - tokenizers, stop words, etc.

`ptfidf` (probabilistic tf-idf) is a python package that implements a generative model for sparse binary token vectors that allows to reap the benefits of tf-idf scores, in particular efficient sparse matrix operations and reasonable results even without training data (scores are similar to tf-idf when there before training).
In addition, it offers the capability of supervised and unsupervised model training for domain adaptation.

# Status

Currently under construction. Working examples and some explanations can be found in [notebooks/examples.ipynb](./notebooks/examples.ipynb).

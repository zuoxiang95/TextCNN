#!/usr/bin/env python
# -*- coding: utf-8 -*-

# configuration
num_classes = 2  # number of label
learning_rate = 0.01
batch_size = 512
decay_steps = 5000  # how many steps before decay learning rate.
decay_rate = 0.9
log_dir = '../TextCNN/log-dir'  # checkpoint location for the model
max_sentence_len = 100  # max sentence length
use_embedding = True  # whether to use embedding or not.
embed_size = 100  # embedding size

num_epochs = 20  # number of epochs to run
validate_every = 5  # Validate every validate_every epochs.
training_data_path = '../TextCNN/data-dir'  # path of training data.

word2vec_model_path = 'train-zhihu4-only-title-all.txt'
num_filters = 256
multi_label_flag = False
filter_sizes = [1, 2, 3, 4, 5, 6, 7]

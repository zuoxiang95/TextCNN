#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class TextCNN(object):

    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size, is_training, multi_label_flag=False, clip_gradients=5.0,
                 initializer=tf.random_normal_initializer(stddev=0.1), decay_rate_big=0.50):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients
        self.decay_step = decay_steps
        self.decay_rate = decay_rate

        # add placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y")

        # dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

    def model(self):
        # Embedding
        embedding_table = tf.get_variable(
            'embedding', [self.vocab_size, self.embed_size], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.5)
        )
        embedded_inputs = tf.nn.embedding_lookup(embedding_table, self.input_x)  # [None, sentence_length, embed_size]
        sentence_embedding_expanded = tf.expand_dim(embedded_inputs, -1)  # [None, sentence_length, embed_size, 1]

        # Convolution layer
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("Convolution-pooling-%s" % filter_size):
                # get a convolution-filter
                conv_filter = tf.get_variable(
                    "filter-%s" % i, [filter_size, self.embed_size, self.num_filters],
                    initializer=tf.truncated_normal_initializer(stddev=0.5)
                )
                conv = tf.nn.conv2d(sentence_embedding_expanded, conv_filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")










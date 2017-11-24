#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, init_learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size, is_training, multi_label_flag=False, clip_gradients=5.0,
                 initializer=tf.random_normal_initializer(stddev=0.1), decay_rate_big=0.50):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.init_learning_rate = tf.Variable(init_learning_rate, trainable=False,
                                              name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.init_learning_rate, self.init_learning_rate * decay_rate_big)
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
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")

        # dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.dense = self.model(self.is_training)
        self.loss = self.add_loss()
        self.train_op = self.add_optimizer(self.global_step)
        self.predictions = tf.arg_max(self.dense, 1, name="predictions")
        self.accuracy = tf.constant(0.5)

    def model(self, is_training):
        # Embedding
        embedding_table = tf.get_variable(
            'embedding', [self.vocab_size, self.embed_size], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.5)
        )
        embedded_inputs = tf.nn.embedding_lookup(embedding_table, self.input_x)  # [None, sentence_length, embed_size]
        sentence_embedding_expanded = tf.expand_dims(embedded_inputs, -1)  # [None, sentence_length, embed_size, 1]

        poolout_list = []
        # Convolution layer
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("Convolution-pooling-%s" % filter_size):
                # get a convolution-filter
                conv_filter = tf.get_variable(
                    "filter-%s" % i, [filter_size, self.embed_size, 1, self.num_filters],
                    initializer=tf.truncated_normal_initializer(stddev=0.5)
                )
                # convolution action
                conv = tf.nn.conv2d(sentence_embedding_expanded, conv_filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                # bias
                b = tf.get_variable("b-%s" % i, [self.num_filters])
                # activation
                conv_out = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                # batch normalization
                # bn = tf.layers.batch_normalization(conv_out, training=is_training)

                # max pooling
                maxpool_out = tf.nn.max_pool(conv_out, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                             strides=[1, 1, 1, 1], padding="VALID", name="max_pool")
                poolout_list.append(maxpool_out)

        # concat the output of max pooling
        pooling_out = tf.concat(poolout_list, 3)
        # reshape the pooling out , shape should be [Batch_size, num_filters_total]
        pooling_out = tf.reshape(pooling_out, [-1, self.num_filters_total])

        # drop out
        dropout_rate = self.dropout_keep_prob if is_training else 0.0
        x = tf.layers.dropout(pooling_out, rate=dropout_rate, name="dropout")

        # dense layer
        dense = tf.layers.dense(x, units=self.num_classes, activation=tf.nn.relu, name="dense")
        return dense

    def add_loss(self, l2_lambda=0.001):
        with tf.variable_scope('loss') as scope:
            print(self.input_y)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.dense)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def add_optimizer(self, global_step):
        with tf.variable_scope('optimizer'):
            learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step, self.decay_step,
                                                       self.decay_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_gradients)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
        return optimize

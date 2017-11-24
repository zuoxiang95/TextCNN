# !/usr/bin/env python
# -*- coding; utf-8 -*-

import os
import pickle
import word2vec

import numpy as np
import tensorflow as tf
import hyperparams as hp

from model import TextCNN
from tflearn.data_utils import pad_sequences
from utils import load_data_multilabel_new, create_voabulary, create_voabulary_label


def main():
    # create vocabulary
    vocabulary_word2index, vocabulary_index2word = create_voabulary(hp.word2vec_model_path, name_scope="cnn2")
    vocab_size = len(vocabulary_index2word)
    print("CNN model's vocabulary size: ", vocab_size)

    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="cnn2")
    train, test, _ = load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label,
                                              multi_label_flag=hp.multi_label_flag,
                                              traning_data_path=hp.training_data_path)
    trainX, trainY = train
    testX, testY = test

    # data processing
    print("Start padding & transform word to one hot vector.")
    trainX = pad_sequences(trainX, maxlen=hp.max_sentence_len, value=0.)
    testX = pad_sequences(testX, maxlen=hp.max_sentence_len, value=0.)
    print("After padding, input data looks like: ", trainX[0])
    print("Padding finish!")

    # create a Session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    is_training = True
    with tf.Session(config=config) as sess:
        # Initial model
        textCNN = TextCNN(hp.filter_sizes, hp.num_filters, hp.num_classes, hp.learning_rate, hp.batch_size,
                          hp.decay_steps, hp.decay_rate, hp.max_sentence_len, vocab_size, hp.embed_size, is_training,
                          multi_label_flag=hp.multi_label_flag)
        saver = tf.train.Saver()
        if os.path.exists(hp.log_dir):
            print("Restore Variable from Checkpoint")
            saver.recover_last_checkpoints(hp.log_dir)
        else:
            print("Initialize Variables.")
            sess.run(tf.global_variables_initializer())
            if hp.use_embedding:
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN,
                                                 word2vec_model_path=hp.word2vec_model_path)
        current_epoch = sess.run(textCNN.epoch_step)

        # feeding data
        num_data = len(trainX)
        batch_size = hp.batch_size
        for epoch in range(current_epoch, hp.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, num_data, batch_size), range(batch_size, num_data, batch_size)):
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])  # ;print("trainY[start:end]:",trainY[start:end])
                feed_dict = {textCNN.input_x: trainX[start:end], textCNN.dropout_keep_prob: 0.5}
                if not hp.multi_label_flag:
                    feed_dict[textCNN.input_y] = trainY[start:end]
                else:
                    feed_dict[textCNN.input_y_multilabel] = trainY[start:end]
                curr_loss, curr_acc, _ = sess.run([textCNN.loss_val, textCNN.accuracy, textCNN.train_op],
                                                  feed_dict)  # curr_acc--->TextCNN.accuracy
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                        epoch, counter, loss / float(counter),
                        acc / float(counter)))  # tTrain Accuracy:%.3f---》acc/float(counter)

                    # epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch, hp.validate_every, (epoch % hp.validate_every == 0))
            if epoch % hp.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, textCNN, testX, testY, batch_size, vocabulary_index2word_label)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                # save model to checkpoint
                save_path = hp.log_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:", word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(hp.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, hp.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textCNN, evalX, evalY, batch_size, vocabulary_index2word_label):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        if not hp.multi_label_flag:
            feed_dict[textCNN.input_y] = evalY[start:end]
        else:
            feed_dict[textCNN.input_y_multilabel] = evalY[start:end]
        curr_eval_loss, logits, curr_eval_acc = sess.run([textCNN.loss_val, textCNN.logits, textCNN.accuracy],
                                                         feed_dict)  # curr_eval_acc--->textCNN.accuracy
        # label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        # curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)


# 从logits中取出前五 get label using logits
def get_label_using_logits(logits, vocabulary_index2word_label, top_number=1):
    # print("get_label_using_logits.logits:",logits) #1-d array: array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list = np.argsort(logits)[-top_number:]
    index_list = index_list[::-1]
    # label_list=[]
    # for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list


# 统计预测的准确率
def calculate_accuracy(labels_predicted, labels, eval_counter):
    label_nozero = []
    # print("labels:",labels)
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    if eval_counter < 2:
        print("labels_predicted:", labels_predicted, " ;labels_nozero:", label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

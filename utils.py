# -*- coding: utf-8 -*-
import codecs
import numpy as np
# load data of zhihu
import word2vec
import os
import pickle

PAD_ID = 0

_GO = "_GO"
_END = "_END"
_PAD = "_PAD"


def create_voabulary(simple=None, word2vec_model_path='zhihu-word2vec-title-desc.bin-100',
                     name_scope=''):  # zhihu-word2vec-multilabel.bin-100
    cache_path = 'cache_vocabulary_label_pik/' + name_scope + "_word_voabulary.pik"
    print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))
    if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
        with open(cache_path, 'r') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        if simple is not None:
            word2vec_model_path = 'zhihu-word2vec.bin-100'
        print("create vocabulary. word2vec_model_path:", word2vec_model_path)
        model = word2vec.load(word2vec_model_path, kind='bin')
        vocabulary_word2index['PAD_ID'] = 0
        vocabulary_index2word[0] = 'PAD_ID'
        special_index = 0
        if 'biLstmTextRelation' in name_scope:
            vocabulary_word2index[
                'EOS'] = 1  # a special token for biLstTextRelation model. which is used between two sentences.
            vocabulary_index2word[1] = 'EOS'
            special_index = 1
        for i, vocab in enumerate(model.vocab):
            vocabulary_word2index[vocab] = i + 1 + special_index
            vocabulary_index2word[i + 1 + special_index] = vocab

        # save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):  # 如果不存在写到缓存文件中
            with open(cache_path, 'a') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
    return vocabulary_word2index, vocabulary_index2word


# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_voabulary_label(voabulary_label='train-zhihu4-only-title-all.txt', name_scope='',
                           use_seq2seq=False):  # 'train-zhihu.txt'
    print("create_voabulary_label_sorted.started.traning_data_path:", voabulary_label)
    cache_path = 'cache_vocabulary_label_pik/' + name_scope + "_label_voabulary.pik"
    if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
        with open(cache_path, 'r') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label = pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        zhihu_f_train = codecs.open(voabulary_label, 'r', 'utf8')
        lines = zhihu_f_train.readlines()
        count = 0
        vocabulary_word2index_label = {}
        vocabulary_index2word_label = {}
        vocabulary_label_count_dict = {}  # {label:count}
        for i, line in enumerate(lines):
            if '__label__' in line:  # '__label__-2051131023989903826
                label = line[line.index('__label__') + len('__label__'):].strip().replace("\n", "")
                if vocabulary_label_count_dict.get(label, None) is not None:
                    vocabulary_label_count_dict[label] = vocabulary_label_count_dict[label] + 1
                else:
                    vocabulary_label_count_dict[label] = 1
        list_label = sort_by_value(vocabulary_label_count_dict)

        print("length of list_label:", len(list_label));  # print(";list_label:",list_label)
        countt = 0

        ##########################################################################################
        if use_seq2seq:  # if used for seq2seq model,insert two special label(token):_GO AND _END
            i_list = [0, 1, 2];
            label_special_list = [_GO, _END, _PAD]
            for i, label in zip(i_list, label_special_list):
                vocabulary_word2index_label[label] = i
                vocabulary_index2word_label[i] = label
        #########################################################################################
        for i, label in enumerate(list_label):
            if i < 10:
                count_value = vocabulary_label_count_dict[label]
                print("label:", label, "count_value:", count_value)
                countt = countt + count_value
            indexx = i + 3 if use_seq2seq else i
            vocabulary_word2index_label[label] = indexx
            vocabulary_index2word_label[indexx] = label
        print("count top10:", countt)

        # save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):  # 如果不存在写到缓存文件中
            with open(cache_path, 'a') as data_f:
                pickle.dump((vocabulary_word2index_label, vocabulary_index2word_label), data_f)
    print("create_voabulary_label_sorted.ended.len of vocabulary_label:", len(vocabulary_index2word_label))
    return vocabulary_word2index_label, vocabulary_index2word_label


def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]


def load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label, valid_portion=0.05,
                             max_training_data=1000000,
                             traning_data_path='train-zhihu4-only-title-all.txt', multi_label_flag=True,
                             use_seq2seq=False, seq2seq_label_length=6):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    print("load_data_multilabel_new.training_data_path:", traning_data_path)
    zhihu_f = codecs.open(traning_data_path, 'r', 'utf8')  # -zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_decoder_input = []  # ADD 2017-06-15
    for i, line in enumerate(lines):
        x, y = line.split('__label__')  # x='w17314 w5521 w7729 w767 w10147 w111'
        y = y.strip().replace('\n', '')
        x = x.strip()
        if i < 1:
            print(i, "x0:", x)  # get raw x
        # x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        x = x.split(" ")
        x = [vocabulary_word2index.get(e, 0) for e in
             x]  # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        if i < 2:
            print(i, "x1:", x)  # word to index
        if use_seq2seq:  # 1)prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
            ys = y.replace('\n', '').split(" ")  # ys is a list
            _PAD_INDEX = vocabulary_word2index_label[_PAD]
            ys_mulithot_list = [_PAD_INDEX] * seq2seq_label_length  # [3,2,11,14,1]
            ys_decoder_input = [_PAD_INDEX] * seq2seq_label_length
            # below is label.
            for j, y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_mulithot_list[j] = vocabulary_word2index_label[y]
            if len(ys) > seq2seq_label_length - 1:
                ys_mulithot_list[seq2seq_label_length - 1] = vocabulary_word2index_label[_END]  # ADD END TOKEN
            else:
                ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

            # below is input for decoder.
            ys_decoder_input[0] = vocabulary_word2index_label[_GO]
            for j, y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_decoder_input[j + 1] = vocabulary_word2index_label[y]
            if i < 10:
                print(i, "ys:==========>0", ys)
                print(i, "ys_mulithot_list:==============>1", ys_mulithot_list)
                print(i, "ys_decoder_input:==============>2", ys_decoder_input)
        else:
            if multi_label_flag:  # 2)prepare multi-label format for classification
                ys = y.replace('\n', '').split(" ")  # ys is a list
                ys_index = []
                for y in ys:
                    y_index = vocabulary_word2index_label[y]
                    ys_index.append(y_index)
                ys_mulithot_list = transform_multilabel_as_multihot(ys_index)
            else:  # 3)prepare single label format for classification
                ys_mulithot_list = vocabulary_word2index_label[y]
        if i <= 3:
            print("ys_index:")
            # print(ys_index)
            print(i, "y:", y, " ;ys_mulithot_list:", ys_mulithot_list)  # ," ;ys_decoder_input:",ys_decoder_input)
        X.append(x)
        Y.append(ys_mulithot_list)
        if use_seq2seq:
            Y_decoder_input.append(ys_decoder_input)  # decoder input
            # if i>50000:
            #    break
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:", number_examples)  #
    train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    if use_seq2seq:
        train = train + (Y_decoder_input[0:int((1 - valid_portion) * number_examples)],)
        test = test + (Y_decoder_input[int((1 - valid_portion) * number_examples) + 1:],)
    # 5.return
    print("load_data.ended...")
    return train, test, test


# 将LABEL转化为MULTI-HOT
def transform_multilabel_as_multihot(label_list, label_size=1999):  # 1999label_list=[0,1,4,9,5]
    """
    :param label_list: e.g.[0,1,4]
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(label_size)
    # set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

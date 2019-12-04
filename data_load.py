# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/11/25 0025 20:16
# @Author   : yuenobel
# @File     : data_load.py 
# @Software : PyCharm


'''加载数据及批量化数据'''
from __future__ import print_function
from hyperparams import HyperParams as hp
import codecs
import numpy as np
import regex
import tensorflow as tf


def load_de_vocab():
    '''该函数的目的是给德语的每个词分配一个id并返回两个字典，一个是根据词找id，一个是根据id找词。'''
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/de.vocab.tsv', 'r', 'utf-8').
        read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/en.vocab.tsv', 'r', 'utf-8').
        read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents):
    '''该函数一共有两个参数，source_sents和target_sents。可以理解为源语言和目标语言的句子列表。
    每个列表中的一个元素就是一个句子。 首先利用之前定义的两个函数生成双语语言的word/id字典。
    同时遍历这两个参数指示的句子列表。一次遍历一个句子对，在该次遍历中，给每个句子末尾后加一个文本结束符</s>
    </s> 用以表示句子末尾。加上该结束符的句子又被遍历每个词，同时利用双语word/id字典读取word对应的id加入一个新列表中，
    若该word不再字典中则id用1代替（即UNK的id）。如此则生成概率两个用一串id表示的双语句子的列表。
    然后判断这两个句子的长度是否都没超过设定的句子最大长度hp.maxlen,如果没超过，
    则将这两个双语句子id列表加入模型要用的双语句子id列表x_list,y_list中，
    同时将满足最大句子长度的原始句子（用word表示的）也加入到句子列表Sources以及Targets中。
    函数后半部分为Pad操作。关于numpy中的pad操作可以参考numpy–prod和pad运算。这里说该函数的pad运算，
    由于x和y都是一维的，所有只有前后两个方向可以pad，所以pad函数的第二个参数是一个含有两个元素的列表，
    第一个元素为0说明给x或者y前面什么也不pad，即pad上0个数，第二个元素为hp.maxlen-len(x)以及hp.maxlen-len(x)
    代表给x和y后面pad上x和y初始元素个数和句子最大长度差的那么多数值，至于pad成什么数值，
    后面的constant_values给出了，即pad上去的id值为0，这也是我们词汇表中PAD的id。
    经过pad的操作可以保证用id表示的句子列表都是等长的。
    最终返回等长的句子id数组X，Y，以及原始句子李标Sources以及Targets。
    X和Y的shape都为[len(x_list),hp.maxlen]。其中len(x_list)为句子的总个数，hp.maxlen为设定的最大句子长度。'''
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    # index
    x_list, y_list, sources, targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u'</S>').split()]
        y = [en2idx.get(word, 1) for word in (target_sent + u'</S>').split()]
        if max(len(x), len(y)) <= hp.max_seq_len:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            sources.append(source_sent)
            targets.append(target_sent)
    # pad
    X = np.zeros([len(x_list), hp.max_seq_len], np.int32)
    Y = np.zeros([len(y_list), hp.max_seq_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.max_seq_len - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.max_seq_len - len(y)], 'constant', constant_values=(0, 0))
    return X, Y, sources, targets

def load_train_data():
    de_sents = [regex.sub('[^\s\p{Latin}]', '', line) for line in codecs.
        open(hp.source_train, 'r', 'utf-8').read().split('\n') if line and line[0] != '<']
    # 句首为<是数据描述的行，并非真实数据的部分
    en_sents = [regex.sub('[^\s\p{Latin}]', '', line) for line in codecs.
        open(hp.target_train, 'r', 'utf-8').read().split('\n') if line and line[0] != '<']
    X, Y, _, _ = create_data(de_sents, en_sents)
    return X, Y

def load_test_data():
    '''load_test_data和load_train_data类似，区别不大。
    区别在与正在表达式的操作有些许不同，其中用到了一个函数strip(),默认参数的话就是去掉字符串首以及末尾的空白符。
    同时数据文件中每行以"<seg""<seg" 开头的才是真正的训练数据的句子。'''
    def _refine(line):
        line = regex.sub('<[^>]+>', '', line)
        line = regex.sub('[^\s\p{Latin}]', '', line)
        return line.strip()
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').
        read().split('\n') if line and line[:4] == '<seg']
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').
        read().split('\n') if line and line[:4] == '<seg']
    X, Y, sources, targets = create_data(de_sents, en_sents)
    return X, sources, targets

def get_batch_data():
    '''用于依次生产一个batch数据'''
    # Load data
    X, Y = load_train_data()
    # total batch count
    num_batch = len(X) // hp.batch_size
    # convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    # 构建数据集，打散，批量，并丢掉最后一个不够batch_size 的batch
    input_queues = tf.data.Dataset.from_tensor_slices((X, Y))
    input_queues = input_queues.shuffle(1000).batch(hp.batch_size, drop_remainder=True)
    return input_queues  #  shape=([batch_size, max_seq_len], [batch_size, max_seq_len])
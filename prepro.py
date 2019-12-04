# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/11/25 0025 19:47
# @Author   : yuenobel
# @File     : prepro.py.py 
# @Software : PyCharm


'''该文件用于生产源语言和目标文件'''
from __future__ import print_function
from hyperparams import HyperParams as hp
import codecs
import os
import regex
from collections import Counter


def make_vocab(fpath, fname):
    '''
    结构化数据
    :param fpath: 输入文件
    :param fname: 处理后的输出文件
    '''
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub('[^\s\p{Latin}]', '', text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('./preprocessed'):
        os.mkdir('./preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write('{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n'.
                   format('<PAD>', '<UNK>', '<S>', '</S>'))
        for word, cnt in word2cnt.most_common(len(word2cnt)):   # 按照单词出现的频率写入文件
            fout.write('{}\t{}\n'.format(word, cnt))


if __name__ == '__main__':
    make_vocab(hp.source_train, 'de.vocab.tsv')
    make_vocab(hp.source_test, 'en.vocab.tsv')
    print('Done')
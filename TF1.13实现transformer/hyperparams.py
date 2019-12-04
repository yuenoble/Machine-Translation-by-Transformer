# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/11/25 0025 19:27
# @Author   : yuenobel
# @File     : hyperparams.py 
# @Software : PyCharm


'''该文件包含所有需要用到的超参数'''
class HyperParams:
    # data
    source_train = './de-en/train.tags.de-en.de'
    target_train = './de-en/train.tags.de-en.en'
    source_test = 'de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'de-en/IWSLT16.TED.tst2014.de-en.en.xml'

    # training
    batch_size = 32
    lr = 1e-4
    logdir = './logdir'

    # model
    max_seq_len = 20        # 句子的最长长度
    min_cnt = 20            # 单词出现次数太少的，用<UNK>代替
    d_model = 512
    num_layers = 6
    num_heads = 8
    num_epochs = 20
    dropout_rate = 0.1
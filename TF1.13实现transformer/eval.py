# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/11/28 0028 20:08
# @Author   : yuenobel
# @File     : eval.py 
# @Software : PyCharm


from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import HyperParams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import train_Graph
from nltk.translate.bleu_score import corpus_bleu


def eval():
    g = train_Graph(is_training=False)
    print('Graph loaded')
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print('Restored')
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('')[1]  # model name

            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open('results/' + mname, 'w', 'utf-8') as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    x = X[i*hp.batch_size:(i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size:(i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size:(i+1)*hp.batch_size]

                    preds = np.zeros((hp.batch_size, hp.max_seq_len), np.int32)
                    for j in range(hp.max_seq_len):
                        '''每个词每个词地预测。这样，后一个词预测的时候就可以利用前面的信息来解码。
                        所以一共循环hp.max_len次，每次循环用之前的翻译作为解码器的输入翻译的一个词。'''
                        _preds = sess.run(g.preds, {g.x:x, g.y:preds})
                        preds[:, j] = _preds[:, j]

                    for source, target, pred in zip(sources, targets, preds):
                        got = ''.join(idx2en[idx] for idx in pred).split('</S>')[0].strip()
                        fout.write('-source:' + source + '\n')
                        fout.write('-expected:' + target + '\n')
                        fout.write('-got:' + got + '\n\n')
                        fout.flush()

                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
                        score = corpus_bleu(list_of_refs, hypotheses)
                        fout.write('Bleu Score = ' + str(100*score))


if __name__ == '__main__':
    eval()
    print('Done')
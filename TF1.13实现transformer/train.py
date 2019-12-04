# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/11/26 0026 22:14
# @Author   : yuenobel
# @File     : train.py 
# @Software : PyCharm


from __future__ import print_function
import tensorflow as tf
from hyperparams import HyperParams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *
import os, codecs
from tqdm import tqdm


class train_Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()  # shape=[batch_size, max_seq_len]
            else:
                self.x = tf.placeholder(tf.int32, shape=(None, hp.max_seq_len))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.max_seq_len))
            # decoder_inputs
            '''decoder_inputs和self.y相比，去掉了最后一个句子结束符，而在每句话最前面加了一个初始化为2的id，即<S> ，代表开始。'''
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), axis=-1)
            # load_vocab
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()

            # encoder
            with tf.variable_scope('encoder'):
                # input - word embedding
                self.enc = embedding(self.x,
                                     vocab_size=len(de2idx),
                                     d_model=hp.d_model,
                                     scale=True,
                                     scope='enc_embed')
                # input - positional encoding
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.max_seq_len,
                                      d_model=hp.d_model,
                                      zero_pad=False,
                                      scale=False,
                                      scope='enc_pe')
                # Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))
                # 3. num_layers multi-head attention
                for i in range(hp.num_layers):
                    with tf.variable_scope('num_layers_{}'.format(i)):
                        # multi head attention + Add and Norm
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       d_model=hp.d_model,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False)
                        # feed forward + Add and Norm
                        self.enc = feedforward(self.enc, dff=[4*hp.d_model, hp.d_model])

            # decoder
            with tf.variable_scope('decoder'):
                self.dec = embedding(self.decoder_inputs,
                                     vocab_size=len(en2idx),
                                     d_model=hp.d_model,
                                     scale=True,
                                     scope='dec_embed')
                self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.max_seq_len,
                                      d_model=hp.d_model,
                                      zero_pad=False,
                                      scale=False,
                                      scope='dec_pe')
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))
                for i in range(hp.num_layers):
                    with tf.variable_scope('num_layers_{}'.format(i)):
                        # masked multi-head attention
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       d_model=hp.d_model,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope='self-attention')
                        # multi-head attention
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc,
                                                       d_model=hp.d_model,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope='vanilla-attention')
                        self.dec = feedforward(self.dec, dff=[4*hp.d_model, hp.d_model])  # shape=[batch_size, seq_len, d_model]

            # final linear projection
            self.logits = tf.layers.dense(self.dec, len(en2idx)) # shape=[batch_size, seq_len, target_vocab_size]
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1)) # 预测值 shape=[batch_size, seq_len]
            self.istarget = tf.to_float(tf.not_equal(self.y, 0)) # 真实值 shape=[batch_size, seq_len]
            # pad 部分不参与准确率计算
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / tf.reduce_sum(self.istarget)
            tf.summary.scalar('acc', self.acc)

            if is_training:
                # loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                # pad 部分不参与损失计算
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
                # training scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                # summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    de2idx, idx2de = load_de_vocab()
    en2idx, idex2en = load_en_vocab()
    g = train_Graph('train')
    print('Graph loaded')
    # session
    sv = tf.train.Supervisor(
        graph=g.graph,
        logdir=hp.logdir,
        save_model_secs=0
    )
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop():break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_opoch_%02d_gs_%d'%(epoch, gs))
    print('Done')
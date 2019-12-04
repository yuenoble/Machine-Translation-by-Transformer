# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/12/3 0003 19:49
# @Author   : yuenobel
# @File     : train.py 
# @Software : PyCharm


import tensorflow as tf
from hyperparams import HyperParams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from transformer import *
import time
import matplotlib.pyplot as plt


# 动态学习率
'''学习率衰减计算:lrate = d_model^-0.5 * min(step_num^-0.5, step_num*warmup_steps^-1.5)'''
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(0.5) * tf.math.minimum(arg1, arg2)

# 学习率
lr = CustomSchedule(hp.d_model)
# 优化器
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-8)

# temp_learing_rate = CustomSchedule(hp.d_model)
# plt.plot(temp_learing_rate(tf.range(40000, dtype=tf.float32)))
# plt.ylabel('learning rate')
# plt.xlabel('train step')
# plt.show()


# 损失和准确率
'''由于目标序列是填充的，因此在计算损耗时应用填充掩码很重要。 padding的掩码为0，没padding的掩码为1'''
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_fun(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_object(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
# 用于记录损失和准确率
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

de2index, index2de = load_de_vocab()
en2index, index2en = load_en_vocab()
input_vocab_size = len(de2index)
target_vocab_size = len(en2index)

transformer = Transformer(hp.d_model,
                          hp.num_layers,
                          hp.num_heads,
                          hp.dff,
                          input_vocab_size,
                          target_vocab_size,
                          hp.max_seq_len,
                          hp.dropout_rate)

# 创建checkpoint管理器
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          hp.ckpt_path,
                                          max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Load last checkpoint restore')


'''
target分为target_input和target real. target_input是传给解码器的输入，target_real是其左移一个位置的结果，
每个target_input位置对应下一个预测的标签
    如句子=“SOS A丛林中的狮子正在睡觉EOS”
    target_input =“SOS丛林中的狮子正在睡觉”
    target_real =“丛林中的狮子正在睡觉EOS”
transformer是个自动回归模型：它一次预测一个部分，并使用其到目前为止的输出，决定下一步做什么。
在训练期间使用teacher-forcing，即无论模型当前输出什么都强制将正确输出传给下一步。
而预测时则根据前一个的输出预测下一个词
为防止模型在预期输出处达到峰值，模型使用look-ahead mask
'''
@tf.function
def train_step(inputs, targets):
    tar_inp = targets[:, :-1]
    tar_real = targets[:, 1:]
    # 构造mask
    encoder_padding_mask, look_ahead_mask, decoder_padding_mask = create_mask(inputs, tar_inp)

    with tf.GradientTape() as tape:
        pred, _ = transformer(inputs,
                              tar_inp,
                              True,
                              encoder_padding_mask,
                              look_ahead_mask,
                              decoder_padding_mask)
        loss = loss_fun(tar_real, pred)
        # 求梯度
        gradients = tape.gradient(loss, transformer.trainable_variables)
        # 反向传播
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        # 记录loss和acc
        train_loss(loss)
        train_acc(tar_real, pred)


for epoch in range(hp.EPOCHS):
    start_time = time.time()
    # 重置
    train_loss.reset_states()
    train_acc.reset_states()
    for step, (inputs, targets) in enumerate(get_batch_data()):
        print(inputs)
        train_step(inputs, targets)
        if step % 10 == 0:
            print(' epoch{},step:{}, loss:{:.4f}, acc:{:.4f}'.format(
                epoch, step, train_loss.result(), train_acc.result()
            ))
    if epoch % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('epoch{}, save model at {}'.format(epoch, ckpt_save_path))
    print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, train_loss.result(), train_acc.result()))
    print('time in one epoch:{}'.format(time.time() - start_time))

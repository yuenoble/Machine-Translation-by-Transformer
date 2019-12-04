# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/11/30 0030 14:42
# @Author   : yuenobel
# @File     : transformer.py 
# @Software : PyCharm


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(pos, d_model):
    '''
    :param pos: 词在句子中的位置，句子上的维族；（i是d_model上的维度）
    :param d_model: 隐状态的维度，相当于num_units
    :return: 位置编码 shape=[1, position_num, d_model], 其中第一个维度是为了匹配batch_size
    '''
    def get_angles(position, i):
        # 这里的i相当于公式里面的2i或2i+1
        # 返回shape=[position_num, d_model]
        return position / np.power(10000., 2. * (i // 2.) / np.float(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    # 2i位置使用sin编码，2i+1位置使用cos编码
    pe_sin = np.sin(angle_rates[:, 0::2])
    pe_cos = np.cos(angle_rates[:, 1::2])
    pos_encoding = np.concatenate([pe_sin, pe_cos], axis=-1)
    pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], tf.float32)
    return pos_encoding

# # 演示positional_encoding
# pos_encoding = positional_encoding(50, 512)
# print(pos_encoding.shape)
# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()


'''*************** 第一部分: Scaled dot-product attention ***************'''
def scaled_dot_product_attention(q, k, v, mask):
    '''attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V'''
    # query 和 Key相乘
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    scaled_attention =matmul_qk / tf.math.sqrt(dk)
    # 掩码mask
    if mask is not None:
        # 这里将mask的token乘以-1e-9，这样与attention相加后，mask的位置经过softmax后就为0
        # padding位置 mask=1
        scaled_attention += mask * -1e-9
    # 通过softmax获取attention权重, mask部分softmax后为0
    attention_weights = tf.nn.softmax(scaled_attention)  # shape=[batch_size, seq_len_q, seq_len_k]
    # 乘以value
    outputs = tf.matmul(attention_weights, v)  # shape=[batch_size, seq_len_q, depth]
    return outputs, attention_weights

'''*************** 第二部分: Multi-Head Attention ***************'''
'''
multi-head attention包含3部分： - 线性层与分头 - 缩放点积注意力 - 头连接 - 末尾线性层
每个多头注意块有三个输入; Q（查询），K（密钥），V（值）。 它们通过第一层线性层并分成多个头。
注意:点积注意力时需要使用mask， 多头输出需要使用tf.transpose调整各维度。
Q，K和V不是一个单独的注意头，而是分成多个头，因为它允许模型共同参与来自不同表征空间的不同信息。
在拆分之后，每个头部具有降低的维度，总计算成本与具有全维度的单个头部注意力相同。
'''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # d_model必须可以正确分成多个头
        assert d_model % num_heads == 0
        # 分头之后维度
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头，将头个数的维度，放到seq_len前面 x输入shape=[batch_size, seq_len, d_model]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        # 分头前的前向网络，根据q,k,v的输入，计算Q, K, V语义
        q = self.wq(q)  # shape=[batch_size, seq_len_q, d_model]
        k = self.wq(k)
        v = self.wq(v)
        # 分头
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # 通过缩放点积注意力层
        # scaled_attention shape=[batch_size, num_heads, seq_len_q, depth]
        # attention_weights shape=[batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # shape=[batch_size, seq_len_q, num_heads, depth]
        # 把多头合并
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # shape=[batch_size, seq_len_q, d_model]
        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights

# # 测试multi-head attention
# temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
# y = tf.random.uniform((1, 60, 512))
# output, att = temp_mha(y, y, y, None)
# print(output.shape, att.shape)


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x): # x shape=[batch_size, seq_len, d_model]
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

# point wise 前向网络
def point_wise_feed_forward(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation=tf.nn.relu),
        tf.keras.layers.Dense(d_model)
    ])


'''encoder layer:
每个编码层包含以下子层 - Multi-head attention（带掩码） - Point wise feed forward networks
每个子层中都有残差连接，并最后通过一个正则化层。残差连接有助于避免深度网络中的梯度消失问题。 
每个子层输出是LayerNorm(x + Sublayer(x))，规范化是在d_model维的向量上。Transformer一共有n个编码层。
'''
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training, mask):
        # multi head attention (encoder时Q = K = V)
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        output1 = self.layernorm1(inputs + att_output)  # shape=[batch_size, seq_len, d_model]
        # feed forward network
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)  # shape=[batch_size, seq_len, d_model]
        return output2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 input_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.emb = tf.keras.layers.Embedding(input_vocab_size, d_model)  # shape=[batch_size, seq_len, d_model]
        self.pos_encoding = positional_encoding(max_seq_len, d_model)  # shape=[1, max_seq_len, d_model]
        self.encoder_layer = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                              for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training, mask):
        # 输入部分；inputs shape=[batch_size, seq_len]
        seq_len = inputs.shape[1]  # 句子真实长度
        word_embedding = self.emb(inputs)  # shape=[batch_size, seq_len, d_model]
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb= word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x  # shape=[batch_size, seq_len, d_model]

# # 编码器测试
# sample_encoder = Encoder(512, 2, 8, 1024, 5000, 200)
# sample_encoder_output = sample_encoder(tf.random.uniform((64, 120)), False, None)
# print(sample_encoder_output.shape)


# padding mask
def create_padding_mask(seq):
    '''为了避免输入中padding的token对句子语义的影响，需要将padding位mask掉，
    原来为0的padding项的mask输出为1; encoder和decoder过程都会用到'''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 扩充维度以便于使用attention矩阵;seq输入shape=[batch_size, seq_len]；输出shape=[batch_siz, 1, 1, seq_len]
    return seq[:, np.newaxis, np.newaxis, :]

# look-ahead mask
def create_look_ahead_mask(size):
    '''用于对未预测的token进行掩码 这意味着要预测第三个单词，只会使用第一个和第二个单词。
    要预测第四个单词，仅使用第一个，第二个和第三个单词，依此类推。只有decoder过程用到'''
    # 产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # shape=[seq_len, seq_len]

def create_mask(inputs, targets):
    # 编码器只有padding_mask
    encoder_padding_mask = create_padding_mask(inputs)
    # 解码器decoder_padding_mask,用于第二层multi-head attention
    decoder_padding_mask = create_padding_mask(inputs)
    # seq_mask mask掉未预测的词
    seq_mask = create_look_ahead_mask(tf.shape(targets)[1])
    # decoder_targets_padding_mask 解码层的输入padding mask
    decoder_targets_padding_mask = create_padding_mask(targets)
    # 合并解码层mask，用于第一层masked multi-head attention
    look_ahead_mask = tf.maximum(decoder_targets_padding_mask, seq_mask)
    return encoder_padding_mask, look_ahead_mask, decoder_padding_mask

'''
decoder layer:
每个编码层包含以下子层： - Masked muti-head attention（带padding掩码和look-ahead掩码
- Muti-head attention（带padding掩码）value和key来自encoder输出，
query来自Masked muti-head attention层输出 - Point wise feed forward network
每个子层中都有残差连接，并最后通过一个正则化层。残差连接有助于避免深度网络中的梯度消失问题。
每个子层输出是LayerNorm(x + Sublayer(x))，规范化是在d_model维的向量上。Transformer一共有n个解码层。
当Q从解码器的第一个注意块接收输出，并且K接收编码器输出时，注意权重表示基于编码器输出给予解码器输入的重要性。
换句话说，解码器通过查看编码器输出并自我关注其自己的输出来预测下一个字。
ps：因为padding在后面所以look-ahead掩码同时掩padding
'''
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        # masked multi-head attention: Q = K = V
        att_out1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att_out1 = self.dropout1(att_out1, training=training)
        att_out1 = self.layernorm1(inputs + att_out1)
        # multi-head attention: Q=att_out1, K = V = encoder_out
        att_out2, att_weight2 = self.mha2(att_out1, encoder_out, encoder_out, padding_mask)
        att_out2 = self.dropout2(att_out2, training=training)
        att_out2 = self.layernorm2(att_out1 + att_out2)
        # feed forward network
        ffn_out = self.ffn(att_out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        output = self.layernorm3(att_out2 + ffn_out)
        return output, att_weight1, att_weight2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.seq_len = tf.shape
        self.d_model = d_model
        self.num_layers = num_layers
        self.word_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        seq_len = inputs.shape[1]
        attention_weights = {}
        word_embedding = self.word_embedding(inputs)
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x, att1, att2 = self.decoder_layers[i](x, encoder_out, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att1
            attention_weights['decoder_layer{}_att_w2'.format(i + 1)] = att2
        return x, attention_weights

# # 解码器测试
# sample_decoder = Decoder(512, 2, 8, 1024, 5000, 200)
# sample_decoder_output, attn = sample_decoder(tf.random.uniform((64, 100)),
#                                              sample_encoder_output, False, None, None)
# print(sample_decoder_output.shape)
# print(attn['decoder_layer1_att_w2'].shape)


'''Transformer包含编码器、解码器和最后的线性层，解码层的输出经过线性层后得到Transformer的输出'''
class Transformer(tf.keras.Model):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 input_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, dff, input_vocab_size, max_seq_len, dropout_rate)
        self.decoder = Decoder(d_model, num_layers, num_heads, dff, target_vocab_size, max_seq_len, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self, inputs, targets, training, encoder_padding_mask,
             look_ahead_mask, decoder_padding_mask):
        # 首先encoder过程，输出shape=[batch_size, seq_len_input, d_model]
        encoder_output = self.encoder(inputs, training, encoder_padding_mask)
        # 再进行decoder, 输出shape=[batch_size, seq_len_target, d_model]
        decoder_output, att_weights = self.decoder(targets, encoder_output, training,
                                                   look_ahead_mask, decoder_padding_mask)
        # 最后映射到输出层
        final_out = self.final_layer(decoder_output) # shape=[batch_size, seq_len_target, target_vocab_size]
        return final_out, att_weights

# # transformer测试
# sample_transformer = Transformer(
# num_layers=2, d_model=512, num_heads=8, dff=1024,
# input_vocab_size=8500, target_vocab_size=8000, max_seq_len=120
# )
# temp_input = tf.random.uniform((64, 62))
# temp_target = tf.random.uniform((64, 26))
# fn_out, att = sample_transformer(temp_input, temp_target, training=False,
#                               encoder_padding_mask=None,
#                                look_ahead_mask=None,
#                                decoder_padding_mask=None,
#                               )
# print(fn_out.shape)
# print(att['decoder_layer1_att_w1'].shape)
# print(att['decoder_layer1_att_w2'].shape)


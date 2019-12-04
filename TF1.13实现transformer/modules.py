# !/user/bin/env python 
# -*- coding: utf-8 -*- 
# @Time     : 2019/11/26 0026 19:57
# @Author   : yuenobel
# @File     : modules.py 
# @Software : PyCharm


import tensorflow as tf


def normalize(inputs,
              epsilon=1e-8,
              scope='In',
              reuse=None):
    '''
    applies layer normalizition
    :param inputs: 输入，shape=[batch_size, seq_len, d_model]
    :param epsilon: 用于平滑
    :param scope: 命名空间
    :param reuse: 是否重载命名空间的变量
    :return: normalize后的结果，shape和inputs一样
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True) # 均值和方差
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs

def embedding(inputs,
              vocab_size,
              d_model,
              zero_pad=True,
              scale=True,
              scope='embedding',
              reuse=None):
    '''
    embedding
    :param inputs: 输入(id) type=int shape=[batch_size, seq_len]
    :param vacab_size: 单词库单词个数
    :param d_model: num_units
    :param zero_pad: 所有的values第一行是否填充0
    :param scale: 输出是否乘以sqrt(d_model)
    :param scope: 命名空间
    :param reuse: 是否重载命名空间的变量
    :return: shape=[batch_size, seq_len, d_model]
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                  dtype=tf.float32,
                                  shape=[vocab_size, d_model],
                                  initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, d_model]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (d_model ** 0.5)
        return outputs

def multihead_attention(queries,
                        keys,
                        d_model=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope='multihead_attention',
                        reuse=None):
    '''
    applies multi-head attention
    :param queries: shape=[batch_size, seq_len_q, d_model]
    :param keys: shape=[batch_size, seq_len_k, d_model]  keys = values
    :param d_model: num_units
    :param num_heads: number of heads
    :param dropout_rate:
    :param is_training: train or test
    :param causality: 是否mask未来的信息
    :param scope: 命名空间
    :param reuse: 是否重载命名空间的变量
    :return: shape=[batch_size, seq_len_q, d_model]
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if d_model is None:
            d_model = queries.get_shape().as_list[-1]
        # linear projections
        Q = tf.layers.dense(queries, d_model, activation=tf.nn.relu)  # shape=[batch_size, seq_len_q, d_model]
        K = tf.layers.dense(keys, d_model, activation=tf.nn.relu)  # shape=[batch_size, seq_len_k, d_model]
        V = tf.layers.dense(keys, d_model, activation=tf.nn.relu)  # shape=[batch_size, seq_len_v, d_model]
        # 分头
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # shape=[batch_size*num_heads, seq_len_q, d_model/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # shape=[batch_size*num_heads, seq_len_k, d_model/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # shape=[batch_size*num_heads, seq_len_v, d_model/num_heads]
        # multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # shape=[batch_size*num_heads, seq_len_q, seq_len_k]
        # scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # key masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  #转化成0/1表示，0表示pad shape=[[batch_size, seq_len_k]
        key_masks = tf.tile(key_masks, [num_heads, 1])  #复制 shape=[batch_size*num_heads, seq_len_k]
        # 先扩维[batch_size*num_heads, 1, seq_len_k],再复制 shape=[batch_size*num_heads, seq_len_q, seq_len_k]
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        padding = tf.ones_like(outputs) * (-2**32 + 1)
        # padding部分用个极小值代替(mask)
        outputs = tf.where(tf.equal(key_masks, 0), padding, outputs)
        # causality 是否屏蔽未来序列的信息（解码器self attention的时候不能看到自己之后的那些信息）
        '''首先定义一个和outputs后两维的shape相同shape（T_q,T_k）的一个张量（矩阵）。 
        然后将该矩阵转为三角阵tril。三角阵中，对于每一个T_q,凡是那些大于它角标的T_k值全都为0，
        这样作为mask就可以让query只取它之前的key（self attention中query即key）。由于该规律适用于所有query，
        接下来仍用tile扩展堆叠其第一个维度，构成masks，shape为(h*N, T_q,T_k).'''
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # shape=[seq_len_q, seq_len_k]
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # shape=[seq_len_q, seq_len_k]
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # shape=[batch_size*num_heads, seq_len_q, seq_len_k]
            padding = tf.ones_like(masks) * (-2**32 + 1)
            outputs = tf.where(tf.equal(masks, 0), padding, outputs) # shape=[batch_size*num_heads, seq_len_q, seq_len_k]
        outputs = tf.nn.softmax(outputs) # shape=[batch_size*num_heads, seq_len_q, seq_len_k]
        # query masking
        '''所谓要被mask的内容，就是本身不携带信息或者暂时禁止利用其信息的内容。这里query mask也是要将那些初始值为0的queryies
        （比如一开始句子被PAD填充的那些位置作为query） mask住。'''
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # shape=[batch_size, seq_len_q]
        query_masks = tf.tile(query_masks, [num_heads, 1]) # shape=[batch_size*num_heads, seq_len_q]
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # shape=[batch_size*num_heads, seq_len_q, seq_len_k]
        outputs *= query_masks # shape=[batch_size*num_heads, seq_len_q, seq_len_k]
        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_) # shape=[batch_size*num_heads, seq_len_q, d_model/num_heads]
        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # shape=[batch_size, seq_len_q, d_model]
        # 残差连接 Add = (x + sub(x))
        outputs += queries
        # Normalize
        outputs = normalize(outputs) # shape=[batch_size, seq_len_q, d_model]
    return outputs

def feedforward(inputs,
                dff=[2048, 512],
                scope='feed_forward',
                reuse=None):
    '''
    point wise feed forward net
    :param inputs: shape=[batch_size, seq_len, d_model]
    :param dff:
    :param scope:
    :param reuse:
    :return: the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": dff[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": dff[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # add and norm
        outputs += inputs
        outputs = normalize(outputs)
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''对label进行平滑操作:可以看出把之前的one_hot中的0改成了一个很小的数，1改成了一个比较接近于1的数。'''
    k = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / k)
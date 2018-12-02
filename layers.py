# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers


def repeat(x, num, interleave=True):
    """Repeat the elements of a batch `num` times

    Given a batch of elements x = [elem1, elem2, ..., elemN], it repeats each
    element `num` times. If interleave==True, the copies of each element are
    returned consecutively, as in: [elem1, elem1, elem1, ..., elemN, elemN,
    elemN]; otherwise, they are returned as copies of the complete batch, as
    in: [elem1, elem2, ..., elemN, ..., elem1, elem2, ..., elemN].

    The function assumes the first axis in `x` interates over the elements.

    Args:
      x: input tensor, with rank >= 2.
      num: the number of repetitions.
      interleave: whether or not interleave the copies in the batch.

    Returns:
      Output tensor with same rank as `x` and `shape[0] == x.shape[0] * num`.

    """
    if interleave:
        src_shape = x.shape
        tile_repetitions = [1, num] + ([1] * (len(src_shape) - 1))
        dims_reshape = [-1] + list(src_shape[1:])

        x = tf.expand_dims(x, axis=1)
        x = tf.tile(x, tile_repetitions)
        x = tf.reshape(x, dims_reshape)

    else:
        src_shape = x.shape
        tile_repetitions = [num] + ([1] * (len(src_shape) - 1))
        x = tf.tile(x, tile_repetitions)

    return x


def cosine_similarity(a, b, normalize_a=True, normalize_b=True,
                      name='cosine_similarity', epsilon=1e-10):
    """Cosine similarity

    Compute the cosine similarity between vectors in `a` and `b`. It assumes
    tensors `a` and `b` are sequences of vectors stored row-wise. For each
    vector in `a` the cosine similarity is computed with respect to each of the
    vectors in `b`.

    Parameters `normalize_a` and `normalize_b` control whether divide the dot
    product (a·b) by |a| and |b| respectively. The default values compute the
    conventional cosine similarity: (a·b) / (|a||b|).

    Args:
      a: tensor with shape [m, d], containing m d-dimensional vectors.
      b: tensor with shape [n, d], containing n d-dimensional vectors.
      normalize_a: (bool) normalize vectors in `a`.
      normalize_b: (bool) normalize vectors in `b`.
      name: (str) name scope for this layer
      epsilon: (float) value to clip zero norm values.

    Returns:
      A `similarity` tensor with shape [m, n], where similarity[i, j] is the
      cosine similarity between a[i] and b[j].

    """
    with tf.name_scope(name):
        if normalize_a:
            norm_a = tf.expand_dims(tf.norm(a, axis=1), 1)
            a = a / tf.clip_by_value(norm_a, epsilon, float('inf'))

        if normalize_b:
            norm_b = tf.expand_dims(tf.norm(b, axis=1), 1)
            b = b / tf.clip_by_value(norm_b, epsilon, float('inf'))

        sim = tf.matmul(a, b, transpose_b=True, name='cosine_similarity')

    return sim


def glob_max_pool(x, name='glob_max_pool', keep_dims=True, data_format='NHWC'):
    """Global max pooling layer

    Args:
      x: tensor with shape [n, h, w, c] or [n, c, h, w].
      name: (str) name scope for this layer
      keep_dims: (bool) keep original rank or produce a tensor with shape
        [n, c].
      data_format: (str) "NHWC" or "NCHW".

    Returns:
      A tensor of rank 4 or 2 depending on `keep_dims`.

    """
    if data_format == 'NHWC':
        axis = (1, 2)
    elif data_format == 'NCHW':
        axis = (2, 3)
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW"')

    x = tf.reduce_max(x, axis=axis, keep_dims=keep_dims, name=name)
    return x


def glob_avg_pool(x, name='glob_avg_pool', keep_dims=True, data_format='NHWC'):
    """Global average pooling layer

    Args:
      x: tensor with shape [n, h, w, c] or [n, c, h, w].
      name: (str) name scope for this layer
      keep_dims: (bool) keep original rank or produce a tensor with shape
        [n, c].
      data_format: (str) "NHWC" or "NCHW".

    Returns:
      A tensor of rank 4 or 2 depending on `keep_dims`.

    """
    if data_format == 'NHWC':
        axis = (1, 2)
    elif data_format == 'NCHW':
        axis = (2, 3)
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW"')

    x = tf.reduce_mean(x, axis=axis, keep_dims=keep_dims, name=name)
    return x


def film_layer(x, gamma, beta, scale_regularizer=None, noise_distortion=False,
               scope=None, reuse=False, data_format='NHWC'):
    """Feature-wise Linear Modulation Layer

    Reference paper: https://arxiv.org/abs/1709.07871v1

    Apply an affine transformation separately on each feature map. Gamma and
    beta values operate in the "delta regime":

      (gamma + 1) * x + beta

    If scale_regularizer is a function (e.g. tf.contrib.layers.l1_regularizer),
    then the above formula is extended with the regularized gamma0 and beta0
    scaling factors:

      (gamma0 * gamma + 1) * x + (beta0 * beta)

    Additionally, if noise_distortion is True, multiplicative gaussian noise
    gamma_noise, beta_noise is introduced (for an ablation study):

      (gamma_noise * gamma0 * gamma + 1) * x + (beta_noise * beta0 * beta)

    Args:
      x: input tensor with shape [n, h, w, c] or [n, h, w, c].
      gamma: tensor with the scaling factors for the affine
        transformations. Shape [n, c].
      beta: tensor with the translation values for the affine
        transformations. Shape [n, c].
      scale_regularizer: None or a regularizer function. In the second case,
        gamma0 and beta0 are introduced (and regularized by
        `scale_regularizer`) in the FiLM formula.
      noise_distortion: (bool) whether or not introduce gamma_noise and
        beta_noise factors in the FiLM formula.
      scope: variable scope for the learned parameters (gamma0 and beta0). Only
        relevant if `scale_regularizer is not None`.
      reuse: (bool, tf.AUTO_REUSE or None) whether reuse the variable scope or
        not.
      data_format: (str) 'NHWC' or 'NCHW'.

    Returns:
      Output tensor.

    """
    m = gamma.shape[0]  # same shape for beta
    c = gamma.shape[1]

    if data_format == 'NHWC':
        expand_dims = [m, 1, 1, c]
    elif data_format == 'NCHW':
        expand_dims = [m, c, 1, 1]
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW"')

    with tf.variable_scope(scope, default_name='film', reuse=reuse):
        # Expand dimensions to broadcast values
        gamma = tf.reshape(gamma, expand_dims)
        beta = tf.reshape(beta, expand_dims)

        if scale_regularizer:
            scale_shape = [1] + expand_dims[1:]

            gamma0 = tf.get_variable('gamma0', scale_shape,
                                     dtype=tf.float32,
                                     initializer=tf.ones_initializer(),
                                     regularizer=scale_regularizer)
            beta0 = tf.get_variable('beta0', shape=scale_shape,
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer(),
                                    regularizer=scale_regularizer)
        else:
            gamma0 = 1
            beta0 = 1

        if noise_distortion:
            gamma_n = tf.random_normal(gamma.shape, mean=1, stddev=0.3)
            beta_n = tf.random_normal(beta.shape, mean=1, stddev=0.3)
        else:
            gamma_n = 1
            beta_n = 1

        y = (1 + gamma_n * gamma0 * gamma) * x + (beta_n * beta0 * beta)

    return y


def conv_block(x, num_outputs, weights_regularizer=None, scope=None,
               reuse=False, training=False, data_format='NHWC'):
    """Convolution block

    Composed by: 3x3 conv, BN, ReLU, 2x2 max pool.

    Args:
      x: input tensor with shape [n, h, w, c] or [n, c, h, w].
      num_outputs: (int) number of filters in the convolution layer.
      weights_regularizer: regularizer applied to the conv layer.
      scope: (str) variable scope for this block.
      reuse: (bool) whether reuse the variable scope or not.
      training: (bool) whether the block is in training mode or not.
      data_format: (str) 'NHWC' or 'NCHW'.

    Returns:
      Output tensor.

    """
    with tf.variable_scope(scope, default_name='conv', reuse=reuse):
        x = layers.conv2d(x, num_outputs=num_outputs, kernel_size=(3, 3),
                          activation_fn=None,
                          weights_initializer=layers.xavier_initializer(),
                          weights_regularizer=weights_regularizer,
                          biases_initializer=tf.zeros_initializer(),
                          data_format=data_format)

        x = layers.batch_norm(x, center=True, scale=True, is_training=training,
                              data_format=data_format)

        x = tf.nn.relu(x)
        x = layers.max_pool2d(x, kernel_size=(2, 2), stride=2,
                              data_format=data_format)

    return x


def film_generator(x1, x2, num_gammas, num_betas, weights_regularizer=None,
                   scope=None, reuse=False, data_format='NHWC'):
    """FiLM generator employed in the cross-modulation block

    Composed by: Concat, ReLU, Global average pooling, FC.

    Args:
      x1, x2: input tensors of rank 4, with ordering according to
        `data_format`. They represent batches of inner representations.
      num_gammas: (int) number of gammas parameters to produce.
      num_betas: (int) number of beta parameters to produce. Typically
        `num_gammas == num_betas`.
      weights_regularizer: None or regularizer applied to the FC layer.
      scope: (str) variable scope for this block.
      reuse: (bool) whether reuse the variable scope or not.
      data_format: (str) either 'NHWC' or 'NCHW'.

    Returns:
      Tuple (gammas, betas), where gammas is a tensor with shape [num_gammas]
      and betas is a tensor with shape [num_betas].

    """
    with tf.variable_scope(scope, default_name='gen', reuse=reuse):
        x1 = glob_avg_pool(x1, 'glob_pool_x1', data_format=data_format,
                           keep_dims=False)
        x2 = glob_avg_pool(x2, 'glob_pool_x2', data_format=data_format,
                           keep_dims=False)
        x = tf.concat([x1, x2], axis=1)
        x = tf.nn.relu(x)

        x = layers.fully_connected(x, num_gammas + num_betas,
                                   activation_fn=None,
                                   weights_regularizer=weights_regularizer)

        gammas, betas = tf.split(x, [num_gammas, num_betas], axis=1,
                                 name='params')
        return (gammas, betas)


def xmod_block(s, q, num_outputs, conv_regularizer=None, gen_regularizer=None,
               film_scale_regularizer=None, scope=None, reuse=False,
               noise_distortion=False, training=False, data_format='NHWC'):
    """Cross-modulation block

    **NOTE** The function assumes that tensor inputs `s` and `q` has the same
    number of images N, and pairs to modulate are selected one-to-one:

      (s[i], q[i]),  i=1...N

    Args:
      s: tensor representing the support set. Shape [n, h, w, c] or
        [n, c, h, w].
      q: tensor representing the query set. Shape [n, h', w', c] or
        [n, c, h', w'] (i.e. same number of examples and channels as `s`).
      conv_regularizer: the regularizer function to use in the conv layers or
        None.
      gen_regularizer: the regularizer function to use in the FC layer of the
        FiLM generator, or None.
      film_scale_regularizer: when a regularizer function, FiLM layer uses the
        regularized gamma0 and beta0 post-multipliers. (See `film_layer`).
      scope: (str) variable scope for this block.
      reuse: (bool) whether reuse the variable scope or not.
      noise_distortion: (bool) when True, multiplicative gaussian noise is
        added to the FiLM generator prediction.
      training: (bool) whether or not the block is running in training mode
        (for BN layers).
      data_format: (str) either 'NHWC' or 'NCHW'.

    Returns:
      New tuple (s', q') where s'[i] is s[i] modulated in the context of q[i]
      and conversely, q'[i] is q[i] modulated in the context of s[i]. Tensor
      shapes remain equal: shape(s) == shape(s') and shape(q) == shape(q').

    """
    with tf.variable_scope(scope, default_name='film_block', reuse=reuse):

        s = layers.conv2d(s, num_outputs=num_outputs, kernel_size=(3, 3),
                          activation_fn=None,
                          weights_initializer=layers.xavier_initializer(),
                          weights_regularizer=conv_regularizer,
                          biases_initializer=tf.zeros_initializer(),
                          scope='Conv', reuse=tf.AUTO_REUSE,
                          data_format=data_format)
        q = layers.conv2d(q, num_outputs=num_outputs, kernel_size=(3, 3),
                          activation_fn=None,
                          weights_initializer=layers.xavier_initializer(),
                          weights_regularizer=conv_regularizer,
                          biases_initializer=tf.zeros_initializer(),
                          scope='Conv', reuse=tf.AUTO_REUSE,
                          data_format=data_format)

        s = layers.batch_norm(s, center=True, scale=True, is_training=training,
                              scope='BatchNorm', reuse=tf.AUTO_REUSE,
                              data_format=data_format)
        q = layers.batch_norm(q, center=True, scale=True, is_training=training,
                              scope='BatchNorm', reuse=tf.AUTO_REUSE,
                              data_format=data_format)

        gamma_s, beta_s = film_generator(s, q, num_outputs, num_outputs,
                                         weights_regularizer=gen_regularizer,
                                         scope='Generator',
                                         reuse=tf.AUTO_REUSE)
        gamma_q, beta_q = film_generator(q, s, num_outputs, num_outputs,
                                         weights_regularizer=gen_regularizer,
                                         scope='Generator',
                                         reuse=tf.AUTO_REUSE)

        s = film_layer(s, gamma_s, beta_s,
                       scale_regularizer=film_scale_regularizer,
                       noise_distortion=noise_distortion, scope='Film',
                       reuse=tf.AUTO_REUSE, data_format=data_format)
        q = film_layer(q, gamma_q, beta_q,
                       scale_regularizer=film_scale_regularizer,
                       noise_distortion=noise_distortion, scope='Film',
                       reuse=tf.AUTO_REUSE, data_format=data_format)

        s = tf.nn.relu(s)
        q = tf.nn.relu(q)

        s = layers.max_pool2d(s, kernel_size=(2, 2), stride=2,
                              data_format=data_format)
        q = layers.max_pool2d(q, kernel_size=(2, 2), stride=2,
                              data_format=data_format)

    return (s, q)

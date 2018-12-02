# -*- coding: utf-8 -*-
import tensorflow as tf
import layers


def embedding_network(input_tensor, scope=None, reuse=False, training=True,
                      data_format='NHWC'):
    """Embedding network employed in our re-implementation of Matching Networks

    Composed by 4 blocks of (Conv 64x3x3, BN, ReLU, MaxPool 2x2)

    Args:
      input_tensor: tensor representing a batch of images. Shape [n, h, w, c].
      scope: (str) variable scope to use
      reuse: (bool, tf.AUTO_REUSE or None) whether reuse the variable scope or
        not.
      training: (bool) whether the network is in training mode or not
      data_format: (str) 'NHWC' or 'NCHW'.

    Returns:
      Tensor with the d-dimensional embedding for each image. Shape [n, d]

    """
    with tf.variable_scope(scope, 'embedding', reuse=reuse):
        x = layers.conv_block(input_tensor, 64, scope='block0', reuse=reuse,
                              training=training, data_format=data_format)
        x = layers.conv_block(x, 64, scope='block1', reuse=reuse,
                              training=training, data_format=data_format)
        x = layers.conv_block(x, 64, scope='block2', reuse=reuse,
                              training=training, data_format=data_format)
        x = layers.conv_block(x, 64, scope='block3', reuse=reuse,
                              training=training, data_format=data_format)

        x = tf.contrib.layers.flatten(x)

    return x


def matching_network(support_ims, query_ims, num_classes=5, num_shots=1,
                     shared_weights=True, unnormalized_cosine=False,
                     training=True, data_format='NHWC'):
    """Matching Network

    Reference paper: https://arxiv.org/abs/1606.04080v2

    Args:
      support_ims: tensor with the support images. Shape [n, h, w, c] or
        [n, c, h, w], where n = num_classes * num_shots.
      query_ims: tensor with the query images. Shape [m, h, w, c] or
        [m, c, h, w].
      num_classes: (int) number of classes in the support and query batches.
      num_shots: (int) number of samples per class in the support and query
        batches.
      shared_weights: (bool) whether or not use the same embedding function to
        process the support and query images.
      unnormalized_cosine: (bool) whether or not use the unnormalized cosine
        similarity instead of the conventional formulation.
      training: (bool) whether or not the network is running in training mode
        (for BN layers).
      data_format: (str) either 'NHWC' or 'NCHW'.

      **NOTE** It is assumed that `support_ims` are sorted by class.

    Returns:
      Logits tensor with shape [m, num_classes].

    """
    # Obtain embeddings for query and support samples
    scope = 'embedding' if shared_weights else None
    reuse = tf.AUTO_REUSE if shared_weights else False

    q = embedding_network(query_ims, scope=scope, reuse=reuse,
                          training=training, data_format=data_format)
    s = embedding_network(support_ims, scope=scope, reuse=reuse,
                          training=training, data_format=data_format)

    # Cosine similarity
    dist = layers.cosine_similarity(q, s,
                                    normalize_a=(not unnormalized_cosine),
                                    normalize_b=True, name='cosine_similarity')

    # Compute logits adding distances from the same class (num_shots per class)
    n_samples = query_ims.shape[0]
    dist = tf.reshape(dist, [n_samples, num_classes, num_shots])
    logits = tf.reduce_sum(dist, axis=2, name='logits')

    return logits


def cross_modulation_network(support_ims, query_ims, num_classes=5,
                             num_shots=1, film_scale_reg=None, gen_reg=None,
                             conv_reg=None, ablation_study=[], training=True,
                             data_format='NHWC'):
    """Cross-Modulation Network

    Args:
      support_ims: tensor with the support images. Shape [n, h, w, c] or
        [n, c, h, w], where n = num_classes * num_shots.
      query_ims: tensor with the query images, with shape [m, h, w, c] or
        [m, c, h, w].
      num_classes: (int) number of classes in the support and query batches.
      num_shots: (int) number of samples per class in the support and query
        batches.
      film_scale_reg: (None or regularizer) if not None, the FiLM parameters
        are scaled by factors gamma0, beta0, which are regularized by the given
        function.
      gen_reg: (None or regularizer) function to regularize the FC layer in the
        FiLM generator.
      conv_reg: (None or regularizer) function to regularize all the
        convolutional layers in the network (conv blocks and cross-modulation
        blocks).
      ablation_study: (list of int) Introduce multiplicative gaussian noise in
        the FiLM generator prediction. Empty list means no noise introduction;
        possible values in the list are 2 to 4, indicating the block number
        where the generator is distorted with noise.
      training: (bool) whether or not the network is running in training mode
        (for BN layers).
      data_format: (str) either 'NHWC' or 'NCHW'.

    **NOTE** It is assumed that `support_ims` are sorted by class.

    Returns:
      Logits tensor with shape [m, num_classes].

    """
    num_query = query_ims.shape[0]
    num_support = support_ims.shape[0]
    q = query_ims
    s = support_ims
    conv_blocks = range(1)
    filmed_blocks = range(1, 4)

    for i in conv_blocks:
        block_name = 'conv{}'.format(i)

        q = layers.conv_block(q, 64, weights_regularizer=conv_reg,
                              scope=block_name, reuse=tf.AUTO_REUSE,
                              training=training, data_format=data_format)

        s = layers.conv_block(s, 64, weights_regularizer=conv_reg,
                              scope=block_name, reuse=tf.AUTO_REUSE,
                              training=training, data_format=data_format)

    q = layers.repeat(q, num_support, interleave=True)
    s = layers.repeat(s, num_query, interleave=False)

    for i in filmed_blocks:
        block_name = 'conv{}'.format(i)
        s, q = layers.xmod_block(s, q, 64, conv_regularizer=conv_reg,
                                 gen_regularizer=gen_reg,
                                 film_scale_regularizer=film_scale_reg,
                                 scope=block_name, reuse=tf.AUTO_REUSE,
                                 noise_distortion=(i + 1 in ablation_study),
                                 training=training)

    # Unnormalized cosine simmilarity: (s·q) / |s|
    q = tf.contrib.layers.flatten(q)
    s = tf.contrib.layers.flatten(s)
    norm_s = tf.expand_dims(tf.norm(s, axis=-1), 1)
    s = s / tf.clip_by_value(norm_s, 1e-10, float('inf'))
    dot = tf.reduce_sum(q * s, axis=-1, name='dot')

    # Aggregate scores corresponding to support instances of the same class
    x = tf.reshape(dot, [num_query, num_classes, num_shots])
    logits = tf.reduce_sum(x, axis=-1, name='logits')

    return logits

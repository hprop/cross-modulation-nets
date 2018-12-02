from __future__ import print_function
import collections
import json
import os.path

import numpy as np
import tensorflow as tf


BEST_MODEL_PREFIX = 'best_model.ckpt'
EVALUATIONS_FILE = 'evaluations.json'


def categorical_cross_entropy_loss(logits, n_way, q_query):
    """Categorical cross entropy loss

    **NOTE** To generate the labels, the function assumes the query set was
    sorted and grouped by classes. For example, for n_way=5, q_query=2 the
    labels are:

      [class1, class1, class2, class2, ..., class5, class5]

    """
    with tf.name_scope('categorical_xentropy'):
        labels = np.repeat(range(n_way), q_query)
        one_hot = tf.one_hot(labels, n_way)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=one_hot,
                                                       name='loss')
    return tf.reduce_mean(loss)


def adam_optimizer(loss, initial_lr, decay_step, decay_rate):
    """Create op to minimize `loss` with the Adam optimizer

    **NOTE** Internally it uses a global_step variable, required by
    MonitoredTrainingSession.

    """
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(initial_lr, global_step, decay_step,
                                    decay_rate, staircase=True)

    optim = tf.train.AdamOptimizer(learning_rate=lr)
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optim.minimize(loss, global_step=global_step)

    return train_op


def accuracy(logits, n_way, q_query, name=None):
    """Compute the accuracy for the prediction returned by the network

    **NOTE** To generate the labels, the function assumes the query set was
    sorted and grouped by classes. For example, for n_way=5, q_query=2 the
    labels are:

      [class0, class0, class1, class1, ..., class4, class4]

    Returns:
      Same output as tf.metrics.accuracy(): a tuple (accuracy, update_op)

    """
    if name is None:
        name = 'accuracy'

    with tf.name_scope(name):
        labels = np.repeat(range(n_way), q_query)
        pred = tf.argmax(logits, axis=1)
        return tf.metrics.accuracy(labels, pred)


class BestCheckpointSaverHook(tf.train.SessionRunHook):
    """Extend a tensorflow MonitoredSession by tracking the new checkpoints
    generated during training and keeping a copy of the best one in a custom
    directory.

    """
    def __init__(self, save_dir, accuracy_op):
        """Create a BestCheckpointSaverHook

        The hook update 2 files in `saved_dir`:
        - The best checkpoint according to `accuracy_op`. The filename is
          configurable by global parameter `BEST_MODEL_PREFIX`.
        - An evaluations file, with the information of all the checkpoints
        evaluated so far and their accuracy. Its filename is configurable
        through global parameter `EVALUATIONS_FILE`.

        Args:
          save_dir: (str) directory where the checkpoint and evaluations file
            are stored.
          accuracy_op: tensorflow op to compute the accuracy of each model
            checkpoint.

        """
        try:
            self._evals_file = os.path.join(save_dir, EVALUATIONS_FILE)
            f = open(self._evals_file, 'r')
            self._evals = json.load(f)
            f.close()
            self._best_accuracy = max(self._evals, 'accuracy')
        except:
            self._evals = []
            self._best_accuracy = -1

        self._chkpoint_prefix = os.path.join(save_dir, BEST_MODEL_PREFIX)
        self._accuracy_op = accuracy_op
        self._saver = tf.train.Saver(max_to_keep=1)

    def end(self, session):
        chkpoint = collections.OrderedDict()
        chkpoint['iteration'] = int(
            session.run(tf.train.get_or_create_global_step())
        )
        chkpoint['accuracy'] = float(session.run(self._accuracy_op))

        self._evals.append(chkpoint)
        with open(self._evals_file, 'w') as f:
            json.dump(self._evals, f)

        if self._best_accuracy < chkpoint['accuracy']:
            self._saver.save(session, self._chkpoint_prefix,
                             chkpoint['iteration'])
            self._best_accuracy = chkpoint['accuracy']


def restore_best_checkpoint(session, ckpt_dir):
    """Load the best checkpoint saved by a BestCheckpointSaverHook during a
    MonitoredSession.

    **NOTE**: this function depends on the global variables `BEST_MODEL_PREFIX`
    and `EVALUATIONS_FILE` to find the best checkpoint. See more details in
    `BestCheckpointSaverHook` documentation.

    Args:
      session: the tensorflow session where the checkpoint is loaded.
      ckpt_dir: (str) directory where BestCheckpointSaverHook saved the
        checkpoint.

    """
    try:
        evals_file = os.path.join(ckpt_dir, EVALUATIONS_FILE)
        f = open(evals_file)
        evals = json.load(f)
        best_ckpt = max(evals, key=lambda e: e['accuracy'])

    except Exception as e:
        raise Exception('Error reading evaluations summary file: {}'.format(e))

    ckpt_file = "{}-{}".format(BEST_MODEL_PREFIX, best_ckpt['iteration'])
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(ckpt_dir, ckpt_file))


def default_config_proto():
    """Return a ConfigProto configuration to not exhaust all the gpu memory"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def set_random_seed(seed):
    """Use the given seed to perform a reproducible experiment

    `seed` must be an integer value. This function must be called before the
    creation of both, the dataset iterator and the model.

    """
    np.random.seed(seed)
    tf.set_random_seed(seed)


def print_trainable_params(scope=None):
    """Print to stdout a summary of the number of trainable parameters for the
    given scope of the default graph. If scope=None, then all the graph
    trainable parameters are consider.

    """
    n_params = 0
    print('name \t| shape \t| num parameters')

    for var in tf.trainable_variables(scope):
        # shape is an array of tf.Dimension
        shape = var.get_shape()
        n_elems = shape.num_elements()
        print(var.name, shape, n_elems)
        n_params += n_elems

    print('Total parameters:', n_params)


def print_film_post_multipliers(session):
    """Print the FiLM post-multipliers (gamma0 and beta0) of the model"""
    graph = tf.get_default_graph()
    filmed_blocks = range(1, 4)

    print("=====================")
    print("FiLM post-multipliers")
    print("=====================")

    for i in filmed_blocks:
        try:
            gamma_name = 'conv{}/Film/gamma0:0'.format(i)
            beta_name = 'conv{}/Film/beta0:0'.format(i)
            gamma = graph.get_tensor_by_name(gamma_name)
            beta = graph.get_tensor_by_name(beta_name)
        except:
            continue

        gamma_values, beta_values = session.run([gamma, beta])
        print(gamma_name, 'mean:', np.mean(gamma_values))
        print(beta_name, 'mean:', np.mean(beta_values))
        print(gamma_name, 'vector:', gamma_values)
        print(beta_name, 'vector:', beta_values)


def print_film_generator_weight_analysis(session):
    """Weight analysis of the FiLM generators

    The weight matrix of the FiLM generator can be decomposed into two
    submatrices, one responsible for cross-modulation and the other one for
    self-modulation. This function compute the average norm for the column
    vectors of each submatrix and print the results on console.

    """
    graph = tf.get_default_graph()
    w2 = graph.get_tensor_by_name("conv1/Generator/fully_connected/weights:0")
    w3 = graph.get_tensor_by_name("conv2/Generator/fully_connected/weights:0")
    w4 = graph.get_tensor_by_name("conv3/Generator/fully_connected/weights:0")

    w2, w3, w4 = session.run([w2, w3, w4])

    w2_self = w2[:64, :]
    w2_cross = w2[64:, :]
    w3_self = w3[:64, :]
    w3_cross = w3[64:, :]
    w4_self = w4[:64, :]
    w4_cross = w4[64:, :]

    self2 = np.linalg.norm(w2_self, axis=0).mean()
    cross2 = np.linalg.norm(w2_cross, axis=0).mean()
    self3 = np.linalg.norm(w3_self, axis=0).mean()
    cross3 = np.linalg.norm(w3_cross, axis=0).mean()
    self4 = np.linalg.norm(w4_self, axis=0).mean()
    cross4 = np.linalg.norm(w4_cross, axis=0).mean()

    print("==============================")
    print("FiLM generator weight analysis")
    print("==============================")
    print("Block2: self-mod = {}, cross-mod = {}".format(self2, cross2))
    print("Block3: self-mod = {}, cross-mod = {}".format(self3, cross3))
    print("Block4: self-mod = {}, cross-mod = {}".format(self4, cross4))

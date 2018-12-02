# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
import os
import argparse

import numpy as np
import tensorflow as tf

import data
import models
import training


def train(args):
    """Train a network with the options set in `args`.

    A new folder `runs/<dataset>/<model>/<YYYYMMDDhhmmss>/train` is created and
    the tensorflow checkpoints and event files are stored inside.

    """
    episodes_per_epoch = 2000
    num_epochs = 500
    training.set_random_seed(23)

    # Dataset
    data_augm = (args.dataset == 'omniglot')
    splits = data.create_iterator(args.dataset, 'train', args.num_classes,
                                  args.num_shots, args.num_queries,
                                  resize_method='resize',
                                  data_augmentation=data_augm)
    X_supp, y_supp, X_query, y_query = splits

    # Model arguments
    if args.model == 'matching_net':
        kwargs = {}
        kwargs['num_classes'] = args.num_classes
        kwargs['num_shots'] = args.num_shots
        kwargs['unnormalized_cosine'] = args.unnormalized_cosine
        kwargs['training'] = True
        kwargs['data_format'] = 'NHWC'
        prediction = models.matching_network(X_supp, X_query, **kwargs)
    else:
        kwargs = {}
        kwargs['num_classes'] = args.num_classes
        kwargs['num_shots'] = args.num_shots
        kwargs['film_scale_reg'] = tf.contrib.layers.l1_regularizer(0.001)
        kwargs['gen_reg'] = None
        kwargs['conv_reg'] = None
        kwargs['training'] = True
        kwargs['data_format'] = 'NHWC'
        prediction = models.cross_modulation_network(X_supp, X_query, **kwargs)

    training.print_trainable_params()

    # Loss
    xentr_loss = training.categorical_cross_entropy_loss(prediction,
                                                         args.num_classes,
                                                         args.num_queries)
    reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
    train_loss = xentr_loss + reg_loss

    # Optimizer
    train_op = training.adam_optimizer(train_loss, 1e-3, 1e5, 0.5)

    # Create metric: accuracy per epoch
    acc, acc_op = training.accuracy(prediction, args.num_classes,
                                    args.num_queries, 'acc')
    acc_initializer = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='acc'))

    # Create metric: accuracy per episode
    ep_acc, ep_acc_op = training.accuracy(prediction, args.num_classes,
                                          args.num_queries, 'episode_acc')
    ep_acc_initializer = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='episode_acc'))

    # Create metric: mean loss per epoch
    mean_loss, mean_loss_op = tf.metrics.mean(train_loss, name='mean_loss')
    mean_loss_initializer = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='mean_loss'))

    # Create a summary for the loss. By default, the op gets stored in the
    # GraphKeys.SUMMARIES collection of ops, which gets picked up by
    # MonitoredTrainingSession when saving summaries to disk.
    tf.summary.scalar('loss', mean_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('logits', tf.reduce_mean(prediction))
    tf.summary.histogram('logits', prediction)

    session_kwargs = {
        'checkpoint_dir': os.path.join(
            'runs', args.dataset, args.model,
            datetime.datetime.now().strftime('%Y%m%d%H%M'), 'train'
        ),
        'save_checkpoint_secs': 300,
        'save_summaries_steps': episodes_per_epoch * 5,
        'log_step_count_steps': 100,
        'config': training.default_config_proto()
    }
    with tf.train.MonitoredTrainingSession(**session_kwargs) as sess:
        for epoch in range(num_epochs):
            for i in range(episodes_per_epoch):
                _, acc, ep_acc, loss, l2, pred = sess.run([train_op, acc_op,
                                                           ep_acc_op,
                                                           mean_loss_op,
                                                           reg_loss,
                                                           prediction])

                if (i % 10) == 0:
                    print('[epoch {} | {}/{}] episode acc: {:0.4f} | '
                          'epoch acc: {:0.4f} | epoch loss: {:0.4f}'.format(
                              epoch, i, episodes_per_epoch, ep_acc, acc, loss))

                if i == episodes_per_epoch - 1:
                    print('[epoch {} | {}/{}] episode prediction:\n{}'.format(
                        epoch, i, episodes_per_epoch, pred))

                sess.run(ep_acc_initializer)

            sess.run([acc_initializer, mean_loss_initializer])


def evaluate(args):
    """Evaluate checkpoints for the current training

    Folder <subdir>/train is monitored for new checkpoints, which are evaluated
    by using the validation split of the given dataset. Two items get updated
    in <subdir>/eval:

    - The best validation checkpoint, with name `best_model.ckpt-EPISODE`,
      where EPISODE is an integer denoting the training iteration.
    - An `evaluations.json` file with all the checkpoints evaluated and their
      accuracy for the validation dataset.

    Args:
      subdir: (str) subdirectory `runs/<dataset>/<model>/<YYYYmmddhhmmss>`
        containing the running training.

    """
    num_episodes = 1000

    # Dataset
    data_augm = (args.dataset == 'omniglot')
    splits = data.create_iterator(args.dataset, 'val', args.num_classes,
                                  args.num_shots, args.num_queries,
                                  resize_method='resize',
                                  data_augmentation=data_augm)
    X_supp, y_supp, X_query, y_query = splits

    # Model arguments
    if args.model == 'matching_net':
        kwargs = {}
        kwargs['num_classes'] = args.num_classes
        kwargs['num_shots'] = args.num_shots
        kwargs['unnormalized_cosine'] = args.unnormalized_cosine
        kwargs['training'] = False
        kwargs['data_format'] = 'NHWC'
        prediction = models.matching_network(X_supp, X_query, **kwargs)
    else:
        kwargs = {}
        kwargs['num_classes'] = args.num_classes
        kwargs['num_shots'] = args.num_shots
        kwargs['film_scale_reg'] = tf.contrib.layers.l1_regularizer(0.001)
        kwargs['gen_reg'] = None
        kwargs['conv_reg'] = None
        kwargs['training'] = False
        kwargs['data_format'] = 'NHWC'
        prediction = models.cross_modulation_network(X_supp, X_query, **kwargs)

    training.print_trainable_params()

    # Loss
    valid_loss = training.categorical_cross_entropy_loss(prediction,
                                                         args.num_classes,
                                                         args.num_queries)
    # Metrics
    valid_acc, valid_acc_op = training.accuracy(prediction, args.num_classes,
                                                args.num_queries, 'valid_acc')
    mean_loss, mean_loss_op = tf.metrics.mean(valid_loss)
    tf.summary.scalar('accuracy', valid_acc)
    tf.summary.scalar('loss', mean_loss)
    tf.summary.scalar('logits', tf.reduce_mean(prediction))
    tf.summary.histogram('logits', prediction)

    # Eval loop
    tf.contrib.training.evaluate_repeatedly(
        checkpoint_dir=os.path.join(args.subdir, 'train'),
        eval_ops=[valid_acc_op, mean_loss_op],
        eval_interval_secs=1,  # low value to avoid losing new checkpoints
        hooks=[tf.contrib.training.StopAfterNEvalsHook(num_episodes),
               tf.contrib.training.SummaryAtEndHook(
                   log_dir=os.path.join(args.subdir, 'eval'),
                   summary_op=tf.summary.merge_all()
               ),
               training.BestCheckpointSaverHook(
                   os.path.join(args.subdir, 'eval'),
                   valid_acc
               )],
        config=training.default_config_proto()
    )


def test(args):
    """Test the best validation checkpoint in an experiment.

    Load the best checkpoint saved during the validation loop and evaluate on
    the test dataset. Mean accuracy and confidence interval are computed.

    Args:
      subdir: subdirectory `runs/<dataset>/<model>/<YYYYmmddhhmmss>` containing
        the training and validation files.

    """
    num_episodes = 1000

    # Dataset
    data_augm = (args.dataset == 'omniglot')
    splits = data.create_iterator(args.dataset, args.partition,
                                  args.num_classes, args.num_shots,
                                  args.num_queries, resize_method='resize',
                                  data_augmentation=data_augm)
    X_supp, y_supp, X_query, y_query = splits

    # Model arguments
    if args.model == 'matching_net':
        kwargs = {}
        kwargs['num_classes'] = args.num_classes
        kwargs['num_shots'] = args.num_shots
        kwargs['unnormalized_cosine'] = args.unnormalized_cosine
        kwargs['training'] = False
        kwargs['data_format'] = 'NHWC'
        prediction = models.matching_network(X_supp, X_query, **kwargs)
    else:
        kwargs = {}
        kwargs['num_classes'] = args.num_classes
        kwargs['num_shots'] = args.num_shots
        kwargs['film_scale_reg'] = tf.contrib.layers.l1_regularizer(0.001)
        kwargs['gen_reg'] = None
        kwargs['conv_reg'] = None
        kwargs['ablation_study'] = args.ablation
        kwargs['training'] = False
        kwargs['data_format'] = 'NHWC'
        prediction = models.cross_modulation_network(X_supp, X_query, **kwargs)

    training.print_trainable_params()

    # Metrics and summary op
    test_acc, test_acc_op = training.accuracy(prediction, args.num_classes,
                                              args.num_queries, 'test_acc')
    test_acc_initializer = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='test_acc'))

    tf.summary.scalar('accuracy', test_acc)
    tf.summary.scalar('logits', tf.reduce_mean(prediction))
    tf.summary.histogram('logits', prediction)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(test_acc_initializer)
        training.restore_best_checkpoint(sess, os.path.join(args.subdir,
                                                            'eval'))
        if args.model == 'crossmod_net':
            training.print_film_generator_weight_analysis(sess)
            training.print_film_post_multipliers(sess)

        writer = tf.summary.FileWriter(os.path.join(args.subdir, 'test'),
                                       sess.graph)
        episode_acc = []
        for i in range(num_episodes):
            acc, summary = sess.run([test_acc_op, summary_op])
            episode_acc.append(acc)
            sess.run(test_acc_initializer)
            print("[{}/{}] episode accuracy: {}".format(i+1, num_episodes,
                                                        acc))

            if i % 10 == 0:
                writer.add_summary(summary, i)

        mean = np.mean(episode_acc)
        z = 1.96
        ci = z * (np.std(episode_acc) / np.sqrt(num_episodes))
        print('Test accuracy: {:.04f}% (+/- {:.04f})'.format(
            mean * 100, ci * 100))


if __name__ == '__main__':
    desc = ("Train, evaluates or test an experiment. The 'train' command "
            "creates the experiment subdir under the run/ directory of the "
            "CWD. Use 'evaluate' in parallel during the training to evaluate "
            "new checkpoints on the validation dataset. Once the training is "
            "done, use the 'test' command to evaluate the best checkpoint on "
            "the test dataset")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('command', choices=['train', 'eval', 'test'],
                        help="Command to execute")

    parser.add_argument('model', choices=['matching_net', 'crossmod_net'],
                        help=("Model type, either matching network or "
                              "cross-modulation network"))

    parser.add_argument('dataset', choices=['omniglot', 'mini_imagenet'],
                        help="Dataset used for the experiment")

    parser.add_argument('subdir', nargs='?', default='',
                        help=("Only for commands 'eval' and 'test'. The path "
                              "to the experiment subdir where to take the "
                              "model (runs/<dataset>/<model>/"
                              "<YYYYmmddhhmmss>)."))

    parser.add_argument('-p', '--partition', nargs='?', default='test',
                        choices=['train', 'val', 'test'],
                        help=("Only for command 'test'. The partition (train, "
                              "val, test) to use with the command "
                              "(defaults to test)."))

    parser.add_argument('-n', '--num_classes', type=int, default=5,
                        help="Number of classes in each episode.")

    parser.add_argument('-k', '--num_shots', type=int, default=1,
                        help=("Number of support examples per class in each "
                              "episode."))

    parser.add_argument('-q', '--num_queries', type=int, default=1,
                        help=("Number of query examples per class in each "
                              "episode"))

    parser.add_argument('--ablation', type=int, nargs='+', default=[],
                        help=("Only for model 'crossmod_net'. List of "
                              "integers separated by spaces indicating "
                              "per-block gaussian noise to introduce in the "
                              "FiLM prediction. Integers allowed: 2, 3, 4."))

    parser.add_argument('--unnormalized_cosine', action='store_true',
                        default=True, help=("Only for model 'matching_net'. "
                                            "Use the unnormalized cosine "
                                            "similarity instead of the "
                                            "conventional formula."))

    args = parser.parse_args()

    # Increase verbosity level
    tf.logging.set_verbosity(tf.logging.INFO)

    if args.command == 'train':
        train(args)
    elif args.command == 'eval' and args.subdir:
        evaluate(args)
    elif args.command == 'test' and args.subdir:
        test(args)
    else:
        parser.print_help()

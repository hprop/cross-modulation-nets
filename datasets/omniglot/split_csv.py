#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage: split_csv.py [-h] path n_train n_valid

Randomly split a csv file into train.csv, validation.csv and test.csv
partitions. The new files are created in the same basedir as `path`. The number
of test examples is the remaining `n_total - (n_train + n_valid)`.

positional arguments:
  path        The source csv file
  n_train     Number of train examples
  n_valid     Number of validation examples

optional arguments:
  -h, --help  show this help message and exit

"""
import os.path
import argparse

import numpy as np
import data


def main(path, n_train, n_valid):
    dirname = os.path.dirname(path)
    dst_train = os.path.join(dirname, 'train.csv')
    dst_valid = os.path.join(dirname, 'val.csv')
    dst_test = os.path.join(dirname, 'test.csv')

    content = data.csv_to_numpy(path)
    labels = content[:, 1]

    # Split classes into train, valid and test subsets
    classes = np.unique(labels)
    np.random.shuffle(classes)
    cl_train, cl_val, cl_test = np.split(classes, [n_train, n_train + n_valid])

    # Get train indices
    train = np.zeros_like(labels, dtype=bool)
    for c in cl_train:
        train[labels == c] = True

    # Get valid indices
    valid = np.zeros_like(labels, dtype=bool)
    for c in cl_val:
        valid[labels == c] = True

    # Get test indices
    test = np.zeros_like(labels, dtype=bool)
    for c in cl_test:
        test[labels == c] = True

    # Save into files
    data.numpy_to_csv(content[train, :], dst_train)
    data.numpy_to_csv(content[valid, :], dst_valid)
    data.numpy_to_csv(content[test, :], dst_test)

    return dst_train, dst_valid, dst_test


if __name__ == '__main__':
    desc = 'Randomly split a csv file into train.csv, validation.csv and '\
           'test.csv partitions. The new files are created in the same '\
           'basedir as `path`. The number of test examples is the remaining '\
           '`n_total - (n_train + n_valid)`.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('path', help='The source csv file')
    parser.add_argument('n_train', help='Number of train examples', type=int)
    parser.add_argument('n_valid', help='Number of validation examples',
                        type=int)

    args = parser.parse_args()
    main(args.path, args.n_train, args.n_valid)

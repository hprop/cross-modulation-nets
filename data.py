# -*- coding: utf-8 -*-
import csv
import os.path

import numpy as np
import tensorflow as tf


basedir = os.path.dirname(__file__)

# Directories for Mini-ImageNet and Omniglot datasets. It is assumed the images
# should be located in an `images` folder inside them.
OMNIGLOT_DIR = os.path.join(basedir, 'datasets', 'omniglot')
MINI_IMAGENET_DIR = os.path.join(basedir, 'datasets', 'mini-imagenet')


def iterate_csv(path, skip_header=False):
    """Create a generator which yields the rows of a csv file stored in `path`.
    The rows are returned as list of strings.

    If `skip_header=True`, the first row is taken as header and discarded.

    """
    with open(path, 'r') as f:
        reader = csv.reader(f)

        if skip_header:
            next(reader)

        for row in reader:
            yield row


def csv_to_numpy(path, skip_header=False):
    """Create a numpy array with the content of a csv file stored in `path`. The
    resulting array has elements of type string, with dtype=object, and as many
    rows and columns as found in the csv.

    If `skip_header=True`, the first row is taken as header and discarded
    before creating the array.

    """
    data = [r for r in iterate_csv(path, skip_header)]
    result = np.array(data, dtype=object)
    return result


def numpy_to_csv(array, dst):
    """Save the content of a string array into a csv file with path `dst`. The
    rows of the resulting file will correspond to those in `array`.

    """
    with open(dst, 'w') as f:
        w = csv.writer(f, delimiter=',', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)

        for row in array:
            w.writerow(row)


def episode_generator(csv_path, n, k, q, skip_header=False):
    """Create a generator which yields episodes for a n-way, k-shot, q-query
    classification task.

    The function reads a csv file with 2-element rows: the relative path of an
    image and the corresponding label. The absolute path is created as:
    <base_dir>/images/<rel_path>, where <base_dir> is the base directory of
    `csv_path` and <rel_path> is the relative path of the image in the csv
    file.

    In each episode two batches are returned, one for the support examples, the
    other one for the query examples. This function does not read and decode
    the images, but only returns the paths and labels for each example in the
    batches.

    Both batches are uniformly sampled and they have no element in common.

    Parameters
    ----------
    csv_path: string
        Path to the csv file with the image paths and labels

    n: int
        Number of sampled classes for each episode (i.e. the N-way)

    k: int
        Number of support samples for each sampled class in an episode
        (i.e. the K-shot)

    q: int
        Number of query samples for each sampled class in an episode (i.e. the
        Q-query)

    skip_header: bool
        If `skip_header=True`, the first row in the input csv file is taken as
        header and discarded before random sampling batches from it.

    Yield
    -----
    Tuple (support_set_paths, support_set_labels, query_set_paths,
    query_set_labels).

    The four elements are numpy 1D arrays of strings (dtype=object), the first
    two having n*k elements, the last two having n*q elements.

    """
    data = csv_to_numpy(csv_path, skip_header=skip_header)
    image_dir = os.path.join(os.path.dirname(csv_path), 'images', '')

    paths = np.repeat(image_dir, data.shape[0]) + data[:, 0]
    labels = data[:, 1]

    classes = np.sort(np.unique(labels))
    table = {c: np.where(labels == c)[0] for c in classes}  # index table

    while True:
        support_set = []
        query_set = []
        choice = np.random.choice(classes, size=n, replace=False)

        for c in choice:
            sample = np.random.choice(table[c], size=k+q, replace=False)
            support_set += sample[:k].tolist()
            query_set += sample[k:].tolist()

        yield (paths[support_set], labels[support_set],
               paths[query_set], labels[query_set])


def create_dataset(csv_path, n, k, q, output_size, resize_method='resize',
                   random_rotations=False, num_channels=3, skip_header=False):
    """Create a tf.data.Dataset object which can be used to return episodes for
    a n-way, k-shot, q-query classification task.

    Internally it uses the `episode_generator` to build the Dataset object.

    Parameters
    ----------
    csv_path: string
        Path to the csv file with the image paths and labels. See
        `episode_generator` function for details about the format required in
        the csv.

    n: int
        Number of sampled classes for each episode (i.e. the N-way)

    k: int
        Number of support samples for each sampled class in an episode
        (i.e. the K-shot)

    q: int
        Number of query samples for each sampled class in an episode (i.e. the
        Q-query)

    output_size: tuple (height, width)
        Size for the output images. Images are resized or cropped to this size
        according to the method specified in `resize_method`.

    resize_method: 'resize' or 'crop'
        Method used to fit the image size to `output_size`. When 'crop', the
        image can be cropped or padded with zeros. Defaults to 'resize'.

    random_rotations: bool
        if True, perform random rotations for each class sampled. The random
        rotations are multiples of 90 degrees.

    num_channels: int
        Number of channels for the images in the dataset. Defaults to 3.

    skip_header: bool
        If `skip_header=True`, the first row in the input csv file is taken as
        header and discarded before processing.

    Return
    ------
    A tf.data.Dataset object which allows iterate over batches of image
    samples. The examples are returned in a tuple of tensors:

    (support_set, support_set_labels, query_set, query_set_labels)

    The shape and type of the tensors are:
    - supppor_set: [n*k, height, width, num_channels], float32.
    - support_set_labels: [n*k], string.
    - query_set: [n*q, height, width, num_channels], float32.
    - query_set_labels: [n*q], string.

    To obtain the iterator from the dataset, use the method
    make_one_shot_iterator():

    >>> dataset = data.create_dataset(csv_path, 5, 1, 3, [28, 28])
    >>> iterator = dataset.make_one_shot_iterator()
    >>> supp_set, supp_set_lab, query_set, query_set_lab = iterator.get_next()

    """
    def read_image(path):
        """Take a tensor with type string representing the path to an image, and
        return the decoded content of the image as a tensor with type uint8
        and shape [height, width, num_channels].

        """
        content = tf.read_file(path)
        decoded = tf.image.decode_image(content, channels=num_channels)
        expanded = tf.expand_dims(decoded, 0)
        expanded.set_shape([1, None, None, 3])  # required for resize_images()

        if resize_method == 'crop':
            final = tf.image.resize_image_with_crop_or_pad(expanded,
                                                           output_size[0],
                                                           output_size[1])
        elif resize_method == 'resize':
            final = tf.image.resize_images(expanded, output_size)

        else:
            raise ValueError('method should be either "resize" or "crop"')

        return tf.to_float(final[0])

    def parse_episode_batches(support_set_paths, support_set_labels,
                              query_set_paths, query_set_labels):
        """Decode the images in the `support_set_paths` and `query_set_paths`,
        and resize them to `im_size=[height, width]`. Return the resized images
        as uint8 tensors with shape [n, height, width, num_channels] and the
        labels unchanged.

        """
        support_set_ims = tf.map_fn(read_image,
                                    support_set_paths,
                                    dtype=tf.float32,
                                    parallel_iterations=1,
                                    back_prop=False)

        query_set_ims = tf.map_fn(read_image,
                                  query_set_paths,
                                  dtype=tf.float32,
                                  parallel_iterations=1,
                                  back_prop=False)

        # Normalize pixel values to range [0, 1]
        support_set_ims = support_set_ims / 255
        query_set_ims = query_set_ims / 255

        return (support_set_ims, support_set_labels,
                query_set_ims, query_set_labels)

    def rotate_images(support_ims, support_labels, query_ims, query_labels):
        support_splits = tf.split(support_ims, n)
        query_splits = tf.split(query_ims, n)
        rot = tf.random_uniform((n,), minval=1, maxval=5,  # interval [1,5)
                                dtype=tf.int32)

        for i in range(n):
            support_splits[i] = tf.map_fn(lambda im: tf.image.rot90(im, k=rot[i]),
                                          support_splits[i],
                                          dtype=tf.float32,
                                          back_prop=False)
            query_splits[i] = tf.map_fn(lambda im: tf.image.rot90(im, k=rot[i]),
                                        query_splits[i],
                                        dtype=tf.float32,
                                        back_prop=False)

        support_ims = tf.concat(support_splits, axis=0)
        query_ims = tf.concat(query_splits, axis=0)

        return (support_ims, support_labels, query_ims, query_labels)

    def create_generator():
        """Convenience function to use our `episode_generator` into the
        tensorflow Dataset API

        """
        return episode_generator(csv_path, n, k, q, skip_header=skip_header)

    dtype = (tf.string, tf.string, tf.string, tf.string)
    shape = (tf.TensorShape([n*k]), tf.TensorShape([n*k]),
             tf.TensorShape([n*q]), tf.TensorShape([n*q]))

    dataset = tf.data.Dataset.from_generator(create_generator, dtype, shape)
    dataset = dataset.map(parse_episode_batches, num_parallel_calls=4)
    if random_rotations:
        dataset = dataset.map(rotate_images, num_parallel_calls=4)
    dataset = dataset.prefetch(100)

    return dataset


def create_iterator(dataset, partition, n, k, q, resize_method='resize',
                    data_augmentation=False):
    """Create a tensorflow iterator for the Omniglot or miniImagenet datasets

    The returned iterator produces episode batches for a N-way K-shot Q-query
    learning task for the given dataset, where images are scaled to shape
    [28, 28, 3] for Omniglot, and [84, 84, 3] for miniImagenet.

    Args:
      dataset: (str) either 'omniglot' or 'mini_imagenet'
      partition: (str) either 'train', 'val' or 'test'.
      n: (int) number of classes retrieved in each episode.
      k: (int) number of samples per class in the support set (X_supp).
      q: (int) number of samples per class in the query set (X_query).
      data_augmentation: (bool) if True and dataset=='omniglot', perform random
        rotations for each class sampled. The rotation angles are multiples of
        90 degrees. Ignored when dataset=='mini_imagenet'.

    Return:
      Tuple (X_supp, y_supp, X_query, y_query) with the iterator ops. Each time
      they are evaluated a new episode batch is returned.

      - X_supp: float32 tensor with shape [n*k, height, width, 3].
      - y_supp: string tensor with shape [n*k].
      - X_query: float32 tensor with shape [n*q, height, width, 3].
      - y_query: string tensor with shape [n*q].

    **NOTE** The samples in X_supp and X_query are grouped by class. For
    example, for n=2, k=2, q=3, we get:

      y_supp = [classX, classX, classY, classY]
      y_query = [classX, classX, classX, classY, classY, classY]

    """
    if partition not in ['train', 'val', 'test']:
        raise ValueError('unknown partition: {}'.format(partition))

    kwargs = {}
    kwargs['resize_method'] = resize_method

    if dataset == 'omniglot':
        csv_path = os.path.join(OMNIGLOT_DIR, '{}.csv'.format(partition))
        output_size = [28, 28]
        kwargs['random_rotations'] = data_augmentation

    elif dataset == 'mini_imagenet':
        csv_path = os.path.join(MINI_IMAGENET_DIR, '{}.csv'.format(partition))
        output_size = [84, 84]
        kwargs['skip_header'] = True

    else:
        raise ValueError('unknown dataset: {}'.format(dataset))

    dataset = create_dataset(csv_path, n, k, q, output_size, **kwargs)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

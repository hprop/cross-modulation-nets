# Cross-Modulation Networks for Few-shot Learning

This repository contains the code for the paper [Cross-Modulation
Networks for Few-shot Learning](), accepted at the [NIPS 2018 Workshop
on Meta-Learning](http://metalearning.ml/2018/).

## Requirements

The code was developed and tested with the following versions:

- Python 2.7
- Tensorflow 1.4
- Numpy 1.14

## Datasets

### Omniglot

Download the [images_background.zip](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip) and [images_evaluation.zip](https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip) from the [Omniglot repository](https://github.com/brendenlake/omniglot). Unzip both files and move all the alphabets folders to `datasets/omniglot/images` (e.g. `datasets/omniglot/images/Alphabet_of_the_Magi`, `datasets/omniglot/images/Angelic`, etc).

The `datasets/omniglot/all.csv` file lists all the images with their
corresponding classes. The `train.csv`, `val.csv` and `test.csv`
splits were generated using the script
`datasets/omniglot/split_csv.py` (see the usage instructions in the
source file).

### Mini-ImageNet

We employ the [dataset splits](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet) from [Ravi and Larochelle](https://openreview.net/pdf?id=rJY0-Kcll). Please, contact Sachin Ravi to obtain the zip file with the images. Unzip the file in the `datasets/mini-imagenet` folder, so that all the images are placed in `dataset/mini-imagenet/images/*.jpg`.

## Usage

Use the `run_experiment.py` script to train, validate and test a
model. A complete description of the command line options can be
printed issuing `python run_experiment.py -h`.

### Training and validation

To train a model, run the following command:

```
python run_experiment.py train <MODEL> <DATASET> -n <NUM_CLASSES> -k <NUM_SHOTS> -q <NUM_QUERIES>
```

where:

- `<MODEL>`: can be `matching_net` or `crossmod_net`.
- `<DATASET>`: can be `omniglot` or `mini_imagenet`.
- `<NUM_CLASSES>`: is the number of classes in the support and query
  sets.
- `<NUM_SHOTS>`: is the number of examples per class in the support
  set.
- `<NUM_QUERIES>`: is the number of examples per class in the query
  set.

After a few seconds, a new folder
`runs/<DATASET>/<MODEL>/<YYYYMMDDhhmm>` is created, and model
checkpoints and event files are stored inside.

Once launched the training, run in parallel the validation process
issuing the following command on another terminal:

```
python run_experiment.py eval <MODEL> <DATASET> <SUBDIR> -n <NUM_CLASSES> -k <NUM_SHOTS> -q <NUM_QUERIES>
```

where `<SUBDIR>` is the `runs/<DATASET>/<MODEL>/<YYYYMMDDhhmm>` folder
created by the training process.


### Test

Once finished the training, run the following command to test the
model:

```
python run_experiment.py test <MODEL> <DATASET> <SUBDIR> -n <NUM_CLASSES> -k <NUM_SHOTS> -q <NUM_QUERIES>
```

When `<MODEL>` is `crossmod_net`, additional information is printed in
the terminal, employed in the paper analyses (distribution of gamma0
and beta0 FiLM post-multipliers and weight matrix in the FiLM
generator).


### Ablation study

When testing a `crossmod_net` model, use the `--ablation` option to
introduce Gaussian multiplicative noise in the FiLM prediction. It
accepts a space-separated list of integers indicating the blocks to
distort (allowed values: 2 to 4). E.g.:

```
python run_experiment.py test crossmod_net <DATASET> <SUBDIR> -n <NUM_CLASSES> -k <NUM_SHOTS> -q <NUM_QUERIES> --ablation 2 3 4
```

#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import os
import numpy as np
import sys
from . import visualize
from . import nlp
import collections
from six.moves import xrange
import six
import re
from six.moves import urllib
from tensorflow.python.platform import gfile
import tarfile
import gzip

## Load dataset functions
def load_mnist_dataset(shape=(-1,784)):
    """自动下载或加载MNIST数据。然后返回50k个训练数据、10k个验证数据和10k个测试数据。

    参数
    ----------
    shape : tuple
        返回数据的特征形状

    例子
    --------
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    """
    pass

def load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False, second=3):
    """CIFAR-10 数据集包含60000个32x32 RGB 图像，共10类，每类
    6000个数据样本. 一共有50000个训练样本，10000个测试样本。

    The dataset is divided into five training batches and one test batch, each with
    10000 images. The test batch contains exactly 1000 randomly-selected images from
    each class. The training batches contain the remaining images in random order,
    but some training batches may contain more images from one class than another.
    Between them, the training batches contain exactly 5000 images from each class.

    参数
    ----------
    shape : tupe
        The shape of digit images: e.g. (-1, 3, 32, 32) , (-1, 32, 32, 3) , (-1, 32*32*3)
    plotable : True, False
        Whether to plot some image examples.
    second : int
        If ``plotable`` is True, ``second`` is the display time.

    例子
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=True)

    注意
    --------
    CIFAR-10 images can only be display without color change under uint8.
    >>> X_train = np.asarray(X_train, dtype=np.uint8)
    >>> plt.ion()
    >>> fig = plt.figure(1232)
    >>> count = 1
    >>> for row in range(10):
    >>>     for col in range(10):
    >>>         a = fig.add_subplot(10, 10, count)
    >>>         plt.imshow(X_train[count-1], interpolation='nearest')
    >>>         plt.gca().xaxis.set_major_locator(plt.NullLocator())    # 不显示刻度(tick)
    >>>         plt.gca().yaxis.set_major_locator(plt.NullLocator())
    >>>         count = count + 1
    >>> plt.draw()
    >>> plt.pause(3)

    References
    ----------
    `CIFAR website <https://www.cs.toronto.edu/~kriz/cifar.html>`_

    `Code download link <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>`_

    `Code references <https://teratail.com/questions/28932>`_
    """
    pass

def load_ptb_dataset():
    """Penn TreeBank (PTB) 数据集在语言建模论文中出现。
    比如 "Empirical Evaluation and Combination of Advanced Language
    Modeling Techniques" 和 "Recurrent Neural Network Regularization"。

    它的训练集包含 929k 个单词, 验证集包含 73k 个单词, 测试集包含 82k 个单词。
    词汇数量为10k。

    In "Recurrent Neural Network Regularization", they trained regularized LSTMs
    of two sizes; these are denoted the medium LSTM and large LSTM. Both LSTMs
    have two layers and are unrolled for 35 steps. They initialize the hidden
    states to zero. They then use the final hidden states of the current
    minibatch as the initial hidden state of the subsequent minibatch
    (successive minibatches sequentially traverse the training set).
    The size of each minibatch is 20.

    The medium LSTM has 650 units per layer and its parameters are initialized
    uniformly in [−0.05, 0.05]. They apply 50% dropout on the non-recurrent
    connections. They train the LSTM for 39 epochs with a learning rate of 1,
    and after 6 epochs they decrease it by a factor of 1.2 after each epoch.
    They clip the norm of the gradients (normalized by minibatch size) at 5.

    The large LSTM has 1500 units per layer and its parameters are initialized
    uniformly in [−0.04, 0.04]. We apply 65% dropout on the non-recurrent
    connections. They train the model for 55 epochs with a learning rate of 1;
    after 14 epochs they start to reduce the learning rate by a factor of 1.15
    after each epoch. They clip the norm of the gradients (normalized by
    minibatch size) at 10.
    
    返回
    -------
    train_data, valid_data, test_data, vocabulary size
    
    例子
    --------
    >>> train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()

    代码参考
    ---------------
    tensorflow.models.rnn.ptb import reader

    下载链接
    ---------------
    `手动下载链接 <http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz>`_
    """
    pass


def load_matt_mahoney_text8_dataset():
    """Download a text file from Matt Mahoney's website
    if not present, and make sure it's the right size.
    Extract the first file enclosed in a zip file as a list of words.
    This dataset can be used for Word Embedding.

    返回
    --------
    word_list : a list
        a list of string (word).\n
        e.g. [.... 'their', 'families', 'who', 'were', 'expelled', 'from', 'jerusalem', ...]

    例子
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> print('Data size', len(words))
    """
    pass

def load_imbd_dataset(path="imdb.pkl", nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):
    """Load IMDB dataset

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_imbd_dataset(
    ...                                 nb_words=20000, test_split=0.2)
    >>> print('X_train.shape', X_train.shape)
    ... (20000,)  [[1, 62, 74, ... 1033, 507, 27],[1, 60, 33, ... 13, 1053, 7]..]
    >>> print('y_train.shape', y_train.shape)
    ... (20000,)  [1 0 0 ..., 1 0 1]

    References
    -----------
    `Modify from keras. <https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py>`_
    """
    pass

def load_nietzsche_dataset():
    """Load Nietzsche dataset.
    Returns a string.
    
    Examples
    --------
    >>> see tutorial_generate_text.py
    >>> words = tl.files.load_nietzsche_dataset()
    >>> words = basic_clean_str(words)
    >>> words = words.split()
    """
    pass

def load_wmt_en_fr_dataset(data_dir="wmt"):
    """It will download English-to-French translation data from the WMT'15
    Website (10^9-French-English corpus), and the 2013 news test from
    the same site as development set.
    Returns the directories of training data and test data.

    Parameters
    ----------
    data_dir : a string
        The directory to store the dataset.

    References
    ----------
    Code modified from /tensorflow/models/rnn/translation/data_utils.py

    Note
    ----
    Usually, it will take a long time to download this dataset.
    """
    pass



## Load and save network
def save_npz(save_dict={}, name='model.npz'):
    """Input parameters and the file name, save parameters into .npz file. Use tl.utils.load_npz() to restore.

    参数
    ----------
    save_dict : a dictionary
        Parameters want to be saved.
    name : a string or None
        The name of the .npz file.

    例子
    --------
    >>> tl.files.save_npz(network.all_params, name='model_test.npz')
    ... File saved to: model_test.npz
    >>> load_params = tl.files.load_npz(name='model_test.npz')
    ... Loading param0, (784, 800)
    ... Loading param1, (800,)
    ... Loading param2, (800, 800)
    ... Loading param3, (800,)
    ... Loading param4, (800, 10)
    ... Loading param5, (10,)
    >>> put parameters into a TLayer network, please see assign_params()

    参考
    ----------
    `用numpy保存字典 <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    pass

def load_npz(path='', name='model.npz'):
    """Load the parameters of a Model saved by tl.files.save_npz().

    参数
    ----------
    path : a string
        Folder path to .npz file.
    name : a string or None
        The name of the .npz file.

    返回
    --------
    params : list
        A list of parameters in order.

    例子
    --------
    See save_npz and assign_params

    参考
    ----------
    `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    pass

def assign_params(sess, params, network):
    """Assign the given parameters to the TLayer network.

    参数
    ----------
    sess : TensorFlow Session
    params : list
        A list of parameters in order.
    network : a :class:`Layer` class
        The network to be assigned

    例子
    --------
    >>> Save your network as follow:
    >>> tl.files.save_npz(network.all_params, name='model_test.npz')
    >>> network.print_params()
    ...
    ... Next time, load and assign your network as follow:
    >>> sess.run(tf.initialize_all_variables()) # re-initialize, then save and assign
    >>> load_params = tl.files.load_npz(name='model_test.npz')
    >>> tl.files.assign_params(sess, load_params, network)
    >>> network.print_params()

    参考
    ----------
    `Assign value to a TensorFlow variable <http://stackoverflow.com/questions/34220532/how-to-assign-value-to-a-tensorflow-variable>`_
    """
    pass



# Load and save variables
def save_any_to_npy(save_dict={}, name='any.npy'):
    """Save variables to .npy file.

    例子
    ---------
    >>> tl.files.save_any_to_npy(save_dict={'data': ['a','b']}, name='test.npy')
    >>> data = tl.files.load_npy_to_any(name='test.npy')
    >>> print(data)
    ... {'data': ['a','b']}
    """
    pass

def load_npy_to_any(path='', name='any.npy'):
    """Load .npy file.

    Examples
    ---------
    see save_any_to_npy()
    """
    pass


# Visualizing npz files
def npz_to_W_pdf(path=None, regx='w1pre_[0-9]+\.(npz)'):
    """Convert the first weight matrix of .npz file to .pdf by using tl.visualize.W().

    Parameters
    ----------
    path : a string or None
        A folder path to npz files.
    regx : a string
        Regx for the file name.

    Examples
    --------
    >>> Convert the first weight matrix of w1_pre...npz file to w1_pre...pdf.
    >>> tl.files.npz_to_W_pdf(path='/Users/.../npz_file/', regx='w1pre_[0-9]+\.(npz)')
    """
    pass

## Helper functions
def load_file_list(path=None, regx='\.npz'):
    """Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : a string or None
        A folder path.
    regx : a string
        The regx of file name.

    Examples
    ----------
    >>> file_list = tl.files.load_file_list(path=None, regx='w1pre_[0-9]+\.(npz)')
    """
    pass

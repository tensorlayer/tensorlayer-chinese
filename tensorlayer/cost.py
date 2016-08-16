#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

## Cost Functions
def cross_entropy(output, target):
    """返回两个分布的交叉熵，在内部实现Softmax。

    参数
    ----------
    output : Tensor变量，网络输出值（不需要是Softmax输出，因为``cross_entropy``在内部会实现Softmax
        一个有这样数据维度的分布: [None, n_feature].
    target : Tensor变量，目标值
        一个有这样数据维度的分布: [None, n_feature].

    例子
    --------
    >>> ce = tf.cost.cross_entropy(y_logits, y_target_logits)

    注意
    --------
    关于 cross-entropy: `wiki <https://en.wikipedia.org/wiki/Cross_entropy>`_.\n
    """
    pass

def mean_squared_error(output, target):
    """Return the cost function of Mean-squre-error of two distributions.

    参数
    ----------
    output : tensorflow variable
        A distribution with shape: [None, n_feature].
    target : tensorflow variable
        A distribution with shape: [None, n_feature].

    """
    pass

## Regularization Functions
def li_regularizer(scale):
  """li regularization removes the neurons of previous layer, `i` represents `inputs`.\n
  Returns a function that can be used to apply group li regularization to weights.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.



  参赛
  ----------
  scale : 浮点值
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  返回
  --------
  A function with signature `li(weights, name=None)` that apply L1 regularization.

  异常
  ------
  ValueError: if scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  pass

def lo_regularizer(scale):
  """lo regularization removes the neurons of current layer, `o` represents `outputs`\n
  Returns a function that can be used to apply group lo regularization to weights.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  -------
  A function with signature `lo(weights, name=None)` that apply Lo regularization.

  Raises
  ------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  pass

def maxnorm_regularizer(scale=1.0):
  """Max-norm regularization returns a function that can be used
  to apply max-norm regularization to weights.
  About max-norm: `wiki <https://en.wikipedia.org/wiki/Matrix_norm#Max_norm>`_.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  ---------
  A function with signature `mn(weights, name=None)` that apply Lo regularization.

  Raises
  --------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  pass

def maxnorm_o_regularizer(scale):
  """Max-norm output regularization removes the neurons of current layer.\n
  Returns a function that can be used to apply max-norm regularization to each column of weight matrix.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  ---------
  A function with signature `mn_o(weights, name=None)` that apply Lo regularization.

  Raises
  ---------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  pass

def maxnorm_i_regularizer(scale):
  """Max-norm input regularization removes the neurons of previous layer.\n
  Returns a function that can be used to apply max-norm regularization to each row of weight matrix.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  ---------
  A function with signature `mn_i(weights, name=None)` that apply Lo regularization.

  Raises
  ---------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  pass





#

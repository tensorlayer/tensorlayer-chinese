API - 损失函数
=======================

为了尽可能地保持TensorLayer的简洁性，我们最小化损失函数的数量。
因此我们鼓励直接使用TensorFlow官方的函数，比如你可以通过
``tf.contrib.layers.l1_regularizer``, ``tf.contrib.layers.l2_regularizer`` and
``tf.contrib.layers.sum_regularizer`` 来实现L1, L2 和 sum 规则化， 参考 `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_。

假设你有一个三层神经网络，那么其模型参数为 ``[W1, b1, W2, b2, W_out, b_out]``，
这时，你可以像下面的例子那样实现对前两个weights矩阵的L2规则化。

.. code-block:: python

  cost = tl.cost.cross_entropy(y, y_)
  cost = cost + tf.contrib.layers.l2_regularizer(0.001)(network.all_params[0]) + tf.contrib.layers.l2_regularizer(0.001)(network.all_params[2])



.. automodule:: tensorlayer.cost

.. autosummary::

   cross_entropy
   mean_squared_error
   li_regularizer
   lo_regularizer
   maxnorm_regularizer
   maxnorm_o_regularizer
   maxnorm_i_regularizer

损失函数
----------------

.. autofunction:: cross_entropy
.. autofunction:: mean_squared_error


规则化函数
--------------------------

.. autofunction:: li_regularizer
.. autofunction:: lo_regularizer
.. autofunction:: maxnorm_regularizer
.. autofunction:: maxnorm_o_regularizer
.. autofunction:: maxnorm_i_regularizer

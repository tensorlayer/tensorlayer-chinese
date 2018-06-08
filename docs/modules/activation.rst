API - 激活函数
==============================

为了尽可能地保持TensorLayer的简洁性，我们最小化激活函数的数量，因此我们鼓励用户直接使用
TensorFlow官方的函数，比如
``tf.nn.relu``, ``tf.nn.relu6``, ``tf.nn.elu``, ``tf.nn.softplus``,
``tf.nn.softsign`` 等等。更多TensorFlow官方激活函数请看
`这里 <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#activation-functions>`_.


自定义激活函数
---------------------------

在TensorLayer中创造自定义激活函数非常简单。

下面的例子实现了把输入乘以2。对于更加复杂的激活函数，你需要用到TensorFlow的API。

.. code-block:: python

  def double_activation(x):
      return x * 2

.. automodule:: tensorlayer.activation

.. autosummary::

   identity
   ramp
   leaky_relu
   leaky_relu6
   leaky_twice_relu6
   swish
   sign
   hard_tanh
   pixel_wise_softmax

Identity
-----------
.. autofunction:: identity

Ramp
-----------
.. autofunction:: ramp

Leaky Relu
-----------
.. autofunction:: leaky_relu

Leaky ReLU6
------------
.. autofunction:: leaky_relu6

Twice Leaky ReLU6
-----------------
.. autofunction:: leaky_twice_relu6


Swish
-----------
.. autofunction:: swish

可求导 Sign
-----------
.. autofunction:: sign

Hard Tanh
-----------
.. autofunction:: hard_tanh

Pixel-wise Softmax
----------------------
.. autofunction:: pixel_wise_softmax

带有参数的激活函数
------------------------------
请见神经网络层。

API - 激活函数
==============================

为了尽可能地保持TensorLayer的简洁性，我们最小化激活函数的数量。因此我们鼓励直接使用
TensorFlow官方的函数，比如
``tf.nn.relu``, ``tf.nn.relu6``, ``tf.nn.elu``, ``tf.nn.softplus``,
``tf.nn.softsign`` 等等。

.. automodule:: tensorlayer.activation

.. autosummary::

   identity
   ramp

更多TensorFlow官方激活产生请看
`这里 <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#activation-functions>`_.

激活函数
---------------------

.. autofunction:: identity
.. autofunction:: ramp

API - 神经网络层
=========================

为了尽可能地保持TensorLayer的简洁性，我们最小化Layer的数量，因此我们鼓励用户直接使用 TensorFlow官方的函数。
例如，我们不提供local response normalization layer，用户可以在 ``Layer.outputs`` 上使用 ``tf.nn.lrn()`` 来实现之。
更多TensorFlow官方函数请看 `这里 <https://www.tensorflow.org/versions/master/api_docs/index.html>`_。

.. automodule:: tensorlayer.layers

.. autosummary::

   Layer
   InputLayer
   Word2vecEmbeddingInputlayer
   EmbeddingInputlayer
   DenseLayer
   ReconLayer
   DropoutLayer
   DropconnectDenseLayer
   Conv2dLayer
   PoolLayer
   RNNLayer
   FlattenLayer
   ConcatLayer
   ReshapeLayer
   EmbeddingAttentionSeq2seqWrapper
   flatten_reshape
   clear_layers_name
   set_name_reuse
   print_all_variables
   initialize_rnn_state


Basic 层
----------------

.. autoclass:: Layer

输入层
----------

.. autoclass:: InputLayer
  :members:

嵌入层+输入层
-----------------

训练Word2vec的层
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Word2vecEmbeddingInputlayer

嵌入层(输入)
^^^^^^^^^^^^^

.. autoclass:: EmbeddingInputlayer

全连接层
----------------

全连接层
^^^^^^^^^^^^

.. autoclass:: DenseLayer

训练Autoencoder的重构层
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ReconLayer
   :members:

噪声层
----------------

Dropout层
^^^^^^^^^^^

.. autoclass:: DropoutLayer

Dropconnect层
^^^^^^^^^^^^^^^^

.. autoclass:: DropconnectDenseLayer

卷积层
--------------------

2D卷积层
^^^^^^^^^^

.. autoclass:: Conv2dLayer


池化层
--------------------

Max或Mean池化层
^^^^^^^^^^^^^^^^

.. autoclass:: PoolLayer

递归层
----------------

可实现任意cell的递归层(LSTM, GRU等)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RNNLayer

形状修改层
----------------

Flatten层
^^^^^^^^^^^

.. autoclass:: FlattenLayer

Concat层
^^^^^^^^^^

.. autoclass:: ConcatLayer

Reshape层
^^^^^^^^^^^

.. autoclass:: ReshapeLayer

包装器(Wrapper)
----------------

嵌入+注意机制+Seq2seq
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EmbeddingAttentionSeq2seqWrapper
  :members:

开发中与待测试
-------------------------

欢迎大家一起开发与测试，每一点贡献都会署名。

3D卷积层
^^^^^^^^^

.. autoclass:: Conv3dLayer

Maxout层
^^^^^^^^^

.. autoclass:: MaxoutLayer

高斯噪声层
^^^^^^^^^^^^

.. autoclass:: GaussianNoiseLayer

双向递归层
^^^^^^^^^^^

.. autoclass:: BidirectionalRNNLayer

辅助函数
----------------

.. autofunction:: flatten_reshape
.. autofunction:: clear_layers_name
.. autofunction:: set_name_reuse
.. autofunction:: print_all_variables
.. autofunction:: initialize_rnn_state

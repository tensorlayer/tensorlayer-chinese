:mod:`tensorlayer.layers`
=========================

To make TensorLayer simple, we minimize the number of layer classes as much as
we can. So we encourage you to use TensorFlow's function.
For example, we do not provide layer for local response normalization, we suggest
you to apply ``tf.nn.lrn`` on ``Layer.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_

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
----------------

.. autoclass:: InputLayer
  :members:

词向量输入层
---------------------------

.. autoclass:: Word2vecEmbeddingInputlayer
.. autoclass:: EmbeddingInputlayer

全连接层
----------------

.. autoclass:: DenseLayer
.. autoclass:: ReconLayer
   :members:

噪声层
----------------

.. autoclass:: DropoutLayer
.. autoclass:: DropconnectDenseLayer

卷积层
--------------------

.. autoclass:: Conv2dLayer


池化层
--------------------

.. autoclass:: PoolLayer

递归层
----------------

.. autoclass:: RNNLayer

修改形状层
----------------

.. autoclass:: FlattenLayer
.. autoclass:: ConcatLayer
.. autoclass:: ReshapeLayer

包装(Wrapper)
----------------

.. autoclass:: EmbeddingAttentionSeq2seqWrapper
  :members:

开发中与待测试
-------------------------

欢迎大家一起开发与测试，每一点贡献都会署名。

.. autoclass:: Conv3dLayer
.. autoclass:: MaxoutLayer
.. autoclass:: GaussianNoiseLayer
.. autoclass:: BidirectionalRNNLayer

辅助函数
----------------

.. autofunction:: flatten_reshape
.. autofunction:: clear_layers_name
.. autofunction:: set_name_reuse
.. autofunction:: print_all_variables
.. autofunction:: initialize_rnn_state

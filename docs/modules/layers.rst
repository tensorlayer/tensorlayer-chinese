API - 神经网络层
=========================


.. -----------------------------------------------------------
..                        Layer List
.. -----------------------------------------------------------

神经网络层列表
---------------------------------

.. automodule:: tensorlayer.layers

.. autosummary::

   Layer

   Input

   OneHot
   Word2vecEmbedding
   Embedding
   AverageEmbedding

   Dense
   Dropout
   GaussianNoise
   DropconnectDense

   UpSampling2d
   DownSampling2d

   Conv1d
   Conv2d
   Conv3d
   DeConv2d
   DeConv3d
   DepthwiseConv2d
   SeparableConv1d
   SeparableConv2d
   DeformableConv2d
   GroupConv2d

   PadLayer
   PoolLayer
   ZeroPad1d
   ZeroPad2d
   ZeroPad3d
   MaxPool1d
   MeanPool1d
   MaxPool2d
   MeanPool2d
   MaxPool3d
   MeanPool3d
   GlobalMaxPool1d
   GlobalMeanPool1d
   GlobalMaxPool2d
   GlobalMeanPool2d
   GlobalMaxPool3d
   GlobalMeanPool3d
   CornerPool2d

   SubpixelConv1d
   SubpixelConv2d

   SpatialTransformer2dAffine
   transformer
   batch_transformer

   BatchNorm
   BatchNorm1d
   BatchNorm2d
   BatchNorm3d
   LocalResponseNorm
   InstanceNorm
   InstanceNorm1d
   InstanceNorm2d
   InstanceNorm3d
   LayerNorm
   GroupNorm
   SwitchNorm

   RNN
   SimpleRNN
   GRURNN
   LSTMRNN
   BiRNN

   retrieve_seq_length_op
   retrieve_seq_length_op2
   retrieve_seq_length_op3
   target_mask_op
  
   Flatten
   Reshape
   Transpose
   Shuffle

   Lambda

   Concat
   Elementwise
   ElementwiseLambda

   ExpandDims
   Tile

   Stack
   UnStack

   Sign
   Scale
   BinaryDense
   BinaryConv2d
   TernaryDense
   TernaryConv2d
   DorefaDense
   DorefaConv2d
   QuantizedDense
   QuantizedDenseWithBN
   QuantizedConv2d
   QuantizedConv2dWithBN

   PRelu
   PRelu6
   PTRelu6

   flatten_reshape
   initialize_rnn_state
   list_remove_repeat

.. -----------------------------------------------------------
..                        Basic Layers
.. -----------------------------------------------------------

层基础类
-----------

.. autoclass:: Layer

.. -----------------------------------------------------------
..                        Input Layer
.. -----------------------------------------------------------

输入层
---------------

普通输入层
^^^^^^^^^^^^^^^^
.. autofunction:: Input

.. -----------------------------------------------------------
..                        Embedding Layers
.. -----------------------------------------------------------

One-hot 输入层
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: OneHot

Word2Vec Embedding 输入层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Word2vecEmbedding

Embedding 输入层
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Embedding

Average Embedding 输入层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AverageEmbedding

.. -----------------------------------------------------------
..                     Activation Layers
.. -----------------------------------------------------------


有参数激活函数层
---------------------------

PReLU 层
^^^^^^^^^^^^^^^^^
.. autoclass:: PRelu


PReLU6 层
^^^^^^^^^^^^^^^^^^
.. autoclass:: PRelu6


PTReLU6 层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: PTRelu6


.. -----------------------------------------------------------
..                  Convolutional Layers
.. -----------------------------------------------------------

卷积层
---------------------

卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

Conv1d
"""""""""""""""""""""
.. autoclass:: Conv1d

Conv2d
"""""""""""""""""""""
.. autoclass:: Conv2d

Conv3d
"""""""""""""""""""""
.. autoclass:: Conv3d

反卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

DeConv2d
"""""""""""""""""""""
.. autoclass:: DeConv2d

DeConv3d
"""""""""""""""""""""
.. autoclass:: DeConv3d


Deformable 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

DeformableConv2d
"""""""""""""""""""""
.. autoclass:: DeformableConv2d


Depthwise 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

DepthwiseConv2d
"""""""""""""""""""""
.. autoclass:: DepthwiseConv2d


Group 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

GroupConv2d
"""""""""""""""""""""
.. autoclass:: GroupConv2d


Separable 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

SeparableConv1d
"""""""""""""""""""""
.. autoclass:: SeparableConv1d

SeparableConv2d
"""""""""""""""""""""
.. autoclass:: SeparableConv2d


SubPixel 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

SubpixelConv1d
"""""""""""""""""""""
.. autoclass:: SubpixelConv1d

SubpixelConv2d
"""""""""""""""""""""
.. autoclass:: SubpixelConv2d


.. -----------------------------------------------------------
..                        Dense Layers
.. -----------------------------------------------------------

全连接层
-------------

全连接层

^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Dense

Drop Connection 全连接层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DropconnectDense


.. -----------------------------------------------------------
..                       Dropout Layer
.. -----------------------------------------------------------

Dropout 层
-------------------
.. autoclass:: Dropout

.. -----------------------------------------------------------
..                        Extend Layers
.. -----------------------------------------------------------

拓展层
-------------------

Expand Dims 层
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ExpandDims


Tile 层
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Tile

.. -----------------------------------------------------------
..                  Image Resampling Layers
.. -----------------------------------------------------------

图像重采样层
-------------------------

2D 上采样层
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UpSampling2d

2D 下采样层
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DownSampling2d

.. -----------------------------------------------------------
..                      Lambda Layer
.. -----------------------------------------------------------

Lambda 层
---------------

普通 Lambda 层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Lambda

逐点 Lambda 层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ElementwiseLambda

.. -----------------------------------------------------------
..                      Merge Layer
.. -----------------------------------------------------------

合并层
---------------

合并连接层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Concat

逐点合并层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Elementwise

.. -----------------------------------------------------------
..                      Noise Layers
.. -----------------------------------------------------------

噪声层
---------------
.. autoclass:: GaussianNoise

.. -----------------------------------------------------------
..                  Normalization Layers
.. -----------------------------------------------------------

标准化层
--------------------

Batch 标准化层
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm

Batch1d 标准化层
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm1d

Batch2d 标准化层
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm2d

Batch3d 标准化层
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm3d

Local Response 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LocalResponseNorm

Instance 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InstanceNorm

Instance1d 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InstanceNorm1d

Instance2d 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InstanceNorm2d

Instance3d 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InstanceNorm3d

Layer 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LayerNorm

Group 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GroupNorm

Switch 标准化层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SwitchNorm

.. -----------------------------------------------------------
..                     Padding Layers
.. -----------------------------------------------------------

填充层
------------------------

填充层 (底层 API)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PadLayer

1D Zero 填充层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad1d

2D Zero 填充层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad2d

3D Zero 填充层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad3d

.. -----------------------------------------------------------
..                     Pooling Layers
.. -----------------------------------------------------------


池化层
------------------------

池化层 (底层 API)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PoolLayer

1D Max 池化层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool1d

1D Mean 池化层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool1d

2D Max 池化层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool2d

2D Mean 池化层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool2d

3D Max 池化层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool3d

3D Mean 池化层
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool3d

1D Global Max 池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool1d

1D Global Mean 池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool1d

2D Global Max 池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool2d

2D Global Mean 池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool2d

3D Global Max 池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool3d

3D Global Mean 池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool3d

2D Corner 池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: CornerPool2d

.. -----------------------------------------------------------
..                    Quantized Layers
.. -----------------------------------------------------------

量化网络层
------------------

这些层目前还是用矩阵实现的运算，未来我们将提供 bit-count 操作，以实现加速。

Sign
^^^^^^^^^^^^^^
.. autoclass:: Sign

Scale
^^^^^^^^^^^^^^
.. autoclass:: Scale

Binary 全连接层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BinaryDense

Binary (De)卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

BinaryConv2d
"""""""""""""""""""""
.. autoclass:: BinaryConv2d

Ternary 全连接层
^^^^^^^^^^^^^^^^^^^^^^^^^^

TernaryDense
"""""""""""""""""""""
.. autoclass:: TernaryDense

Ternary 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

TernaryConv2d
"""""""""""""""""""""
.. autoclass:: TernaryConv2d

DoReFa 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

DorefaConv2d
"""""""""""""""""""""
.. autoclass:: DorefaConv2d

DoReFa 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

DorefaConv2d
"""""""""""""""""""""
.. autoclass:: DorefaConv2d

Quantization 全连接层
^^^^^^^^^^^^^^^^^^^^^^^^^^

QuantizedDense
"""""""""""""""""""""
.. autoclass:: QuantizedDense

QuantizedDenseWithBN 全连接层+批标准化
""""""""""""""""""""""""""""""""""""
.. autoclass:: QuantizedDenseWithBN

Quantization 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantization
"""""""""""""""""""""
.. autoclass:: QuantizedConv2d

QuantizedConv2dWithBN
"""""""""""""""""""""
.. autoclass:: QuantizedConv2dWithBN


.. -----------------------------------------------------------
..                  Recurrent Layers
.. -----------------------------------------------------------

循环层
---------------------

普通循环层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RNN 层
""""""""""""""""""""""""""
.. autoclass:: RNN


基本RNN层 （使用简单循环单元）
"""""""""""""""""""""""""""""""
.. autoclass:: SimpleRNN


基于GRU的RNN层（使用GRU循环单元）
""""""""""""""""""""""""""""""""
.. autoclass:: GRURNN

基于LSTM的RNN层（使用LSTM循环单元）
""""""""""""""""""""""""""""""""
.. autoclass:: LSTMRNN

Bidirectional 层
"""""""""""""""""""""""""""""""""
.. autoclass:: BiRNN


计算步长函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

方法 1
""""""""""""""""""""""""""
.. autofunction:: retrieve_seq_length_op

方法 2
""""""""""""""""""""""""""
.. autofunction:: retrieve_seq_length_op2

方法 3
""""""""""""""""""""""""""
.. autofunction:: retrieve_seq_length_op3


方法 4
"""""""""""""""""""""""""""
.. autofunction:: target_mask_op


.. -----------------------------------------------------------
..                      Shape Layers
.. -----------------------------------------------------------

形状修改层
------------

Flatten 层
^^^^^^^^^^^^^^^
.. autoclass:: Flatten

Reshape 层
^^^^^^^^^^^^^^^
.. autoclass:: Reshape

Transpose 层
^^^^^^^^^^^^^^^^^
.. autoclass:: Transpose

Shuffle 层
^^^^^^^^^^^^^^^^^
.. autoclass:: Shuffle

.. -----------------------------------------------------------
..               Spatial Transformer Layers
.. -----------------------------------------------------------

空间变换层
-----------------------

2D Affine Transformation 层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpatialTransformer2dAffine

2D Affine Transformation 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: transformer

Batch 2D Affine Transformation 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: batch_transformer

.. -----------------------------------------------------------
..                      Stack Layers
.. -----------------------------------------------------------

堆叠层
-------------

堆叠层
^^^^^^^^^^^^^^
.. autoclass:: Stack

反堆叠层
^^^^^^^^^^^^^^^
.. autoclass:: UnStack


.. -----------------------------------------------------------
..                      Helper Functions
.. -----------------------------------------------------------

帮助函数
------------------------

Flatten 函数
^^^^^^^^^^^^^^^^^
.. autofunction:: flatten_reshape

初始化 循环层 状态
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: initialize_rnn_state

去除列表中重复元素
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: list_remove_repeat


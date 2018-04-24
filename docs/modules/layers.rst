API - 神经网络层
=========================

为了尽可能地保持TensorLayer的简洁性，我们最小化Layer的数量，因此我们鼓励用户直接使用 TensorFlow官方的函数。
例如，虽然我们提供local response normalization layer，但用户也可以在 ``network.outputs`` 上使用 ``tf.nn.lrn()`` 来实现之。
更多TensorFlow官方函数请看 `这里 <https://www.tensorflow.org/versions/master/api_docs/index.html>`_。


了解层
---------------

所有TensorLayer层有如下的属性：

 - ``layer.outputs`` : 一个 Tensor，当前层的输出。
 - ``layer.all_params`` : 一列 Tensor, 神经网络每一个参数。
 - ``layer.all_layers`` : 一列 Tensor, 神经网络每一层输出。
 - ``layer.all_drop`` : 一个字典 {placeholder : 浮点数}, 噪声层的概率。

所有TensorLayer层有如下的方法：

 - ``layer.print_params()`` : 打印出神经网络的参数信息（在执行 ``tl.layers.initialize_global_variables(sess)`` 之后）。另外，也可以使用 ``tl.layers.print_all_variables()`` 来打印出所有参数的信息。
 - ``layer.print_layers()`` : 打印出神经网络每一层输出的信息。
 - ``layer.count_params()`` : 打印出神经网络参数的数量。



神经网络的初始化是通过输入层实现的，然后我们可以像下面的代码那样把不同的层堆叠在一起，实现一个完整的神经网络，因此一个神经网络其实就是一个 ``Layer`` 类。
神经网络中最重要的属性有 ``network.all_params``, ``network.all_layers`` 和 ``network.all_drop``.
其中 ``all_params`` 是一个列表(list)，它按顺序保存了指向神经网络参数(variables)的指针，下面的代码定义了一个三层神经网络，则:

``all_params`` = [W1, b1, W2, b2, W_out, b_out]

若需要取出特定的参数，您可以通过 ``network.all_params[2:3]`` 或 ``get_variables_with_name()`` 函数。
然而 ``all_layers`` 也是一个列表(list)，它按顺序保存了指向神经网络每一层输出的指针，在下面的网络中，则：

``all_layers`` = [drop(?,784), relu(?,800), drop(?,800), relu(?,800), drop(?,800)], identity(?,10)]

其中 ``?`` 代表任意batch size都可以。
你可以通过 ``network.print_layers()`` 和 ``network.print_params()`` 打印出每一层输出的信息以及每一个参数的信息。
若想参看神经网络中有多少个参数，则运行 ``network.count_params()`` 。


.. code-block:: python

  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
  y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

  network = tl.layers.InputLayer(x, name='input_layer')
  network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
  network = tl.layers.DenseLayer(network, n_units=800,
                                  act = tf.nn.relu, name='relu1')
  network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
  network = tl.layers.DenseLayer(network, n_units=800,
                                  act = tf.nn.relu, name='relu2')
  network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
  network = tl.layers.DenseLayer(network, n_units=10,
                                  act = tl.activation.identity,
                                  name='output_layer')

  y = network.outputs
  y_op = tf.argmax(tf.nn.softmax(y), 1)

  cost = tl.cost.cross_entropy(y, y_, name='ce')

  train_params = network.all_params

  train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                              epsilon=1e-08, use_locking=False).minimize(cost, var_list = train_params)

  tl.layers.initialize_global_variables(sess)

  network.print_params()
  network.print_layers()

另外，``network.all_drop`` 是一个字典，它保存了噪声层（比如dropout）的 keeping 概率。
在上面定义的神经网络中，它保存了三个dropout层的keeping概率。

因此，在训练时如下启用dropout层。

.. code-block:: python

  feed_dict = {x: X_train_a, y_: y_train_a}
  feed_dict.update( network.all_drop )
  loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
  feed_dict.update( network.all_drop )

在测试时，如下关闭dropout层。

.. code-block:: python

  feed_dict = {x: X_val, y_: y_val}
  feed_dict.update(dp_dict)
  print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
  print("   val acc: %f" % np.mean(y_val ==
                          sess.run(y_op, feed_dict=feed_dict)))

更多细节，请看 MNIST 例子。


自定义层
----------

一个简单的层
^^^^^^^^^^^^^^^

实现一个自定义层，你需要写一个新的Python类，然后实现 ``outputs`` 表达式。

下面的例子实现了把输入乘以2，然后输出。

.. code-block:: python

  class DoubleLayer(Layer):
      def __init__(
          self,
          layer = None,
          name ='double_layer',
      ):
          # 校验名字是否已被使用（不变）
          Layer.__init__(self, layer=layer, name=name)

          # 本层输入是上层的输出（不变）
          self.inputs = layer.outputs

          # 输出信息（自定义部分）
          print("  I am DoubleLayer")

          # 本层的功能实现（自定义部分）
          self.outputs = self.inputs * 2

          # 更新层的参数（自定义部分）
          self.all_layers.append(self.outputs)


你的Dense层
^^^^^^^^^^^^^^^

在创造自定义层之前，我们来看看全连接（Dense）层是如何实现的。
若不存在Weights矩阵和Biases向量时，它新建之，然后通过给定的激活函数计算出 ``outputs`` 。
在最后，作为一个有新参数的层，我们需要把新参数附加到 ``all_params`` 中。


.. code-block:: python

  class MyDenseLayer(Layer):
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.nn.relu,
        name ='simple_dense',
    ):
        # 校验名字是否已被使用（不变）
        Layer.__init__(self, layer=layer, name=name)

        # 本层输入是上层的输出（不变）
        self.inputs = layer.outputs

        # 输出信息（自定义部分）
        print("  MyDenseLayer %s: %d, %s" % (self.name, n_units, act))

        # 本层的功能实现（自定义部分）
        n_in = int(self.inputs._shape[-1])  # 获取上一层输出的数量
        with tf.variable_scope(name) as vs:
            # 新建参数
            W = tf.get_variable(name='W', shape=(n_in, n_units))
            b = tf.get_variable(name='b', shape=(n_units))
            # tensor操作
            self.outputs = act(tf.matmul(self.inputs, W) + b)

        # 更新层的参数（自定义部分）
        self.all_layers.append(self.outputs)
        self.all_params.extend([W, b])

修改预训练行为
^^^^^^^^^^^^^^^

逐层贪婪预训练方法(Greedy layer-wise pretrain)是深度神经网络的初始化非常重要的一种方法，
不过对不同的网络结构和应用，往往有不同的预训练的方法。

例如 `"普通"稀疏自编码器(Vanilla Sparse Autoencoder ) <http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity>`_ 如下面的代码所示，使用 KL divergence 实现（对应于sigmoid)，
但是对于 `深度整流神经网络(Deep Rectifier Network) <http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf>`_ ，
可以通过对神经元输出进行L1规则化来实现稀疏。


.. code-block:: python

  # Vanilla Sparse Autoencoder
  beta = 4
  rho = 0.15
  p_hat = tf.reduce_mean(activation_out, reduction_indices = 0)
  KLD = beta * tf.reduce_sum( rho * tf.log(tf.div(rho, p_hat))
          + (1- rho) * tf.log((1- rho)/ (tf.sub(float(1), p_hat))) )

预训练的方法太多了，出于这个原因，TensorLayer 提供了一种简单的方法来自定义自己的预训练方法。
对于自编码器，TensorLayer 使用 ``ReconLayer.__init__()`` 来定义重构层（reconstruction layer）和损失函数。
要自定义自己的损失函数，只需要在 ``ReconLayer.__init__()`` 中修改 ``self.cost`` 就可以了。
如何写出自己的损失函数，请阅读  `Tensorflow Math <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html>`_ 。
默认情况下， ``重构层(ReconLayer)`` 只使用 ``self.train_params = self.all _params[-4:]`` 来更新前一层的 Weights 和 Biases，这4个参数为 ``[W_encoder，b_encoder，W_decoder，b_decoder]`` ，其中 ``W_encoder，b_encoder`` 属于之前的 Dense 层，  ``W_decoder，b_decoder]`` 属于当前的重构层。
此外，如果您想要同时更新前 2 层的参数，只需要修改 ``[-4:]`` 为 ``[-6:]``。

.. code-block:: python

  ReconLayer.__init__(...):
      ...
      self.train_params = self.all_params[-4:]
      ...
  	self.cost = mse + L1_a + L2_w


层预览表
---------

.. automodule:: tensorlayer.layers

.. autosummary::
   get_variables_with_name
   get_layers_with_name
   set_name_reuse
   print_all_variables
   initialize_global_variables

   Layer

   InputLayer
   OneHotInputLayer
   Word2vecEmbeddingInputlayer
   EmbeddingInputlayer
   AverageEmbeddingInputlayer

   DenseLayer
   ReconLayer
   DropoutLayer
   GaussianNoiseLayer
   DropconnectDenseLayer

   Conv1dLayer
   Conv2dLayer
   DeConv2dLayer
   Conv3dLayer
   DeConv3dLayer
   UpSampling2dLayer
   DownSampling2dLayer
   DeformableConv2dLayer
   AtrousConv1dLayer
   AtrousConv2dLayer

   Conv1d
   Conv2d
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

   SubpixelConv1d
   SubpixelConv2d

   SpatialTransformer2dAffineLayer
   transformer
   batch_transformer

   BatchNormLayer
   LocalResponseNormLayer
   InstanceNormLayer
   LayerNormLayer

   ROIPoolingLayer

   TimeDistributedLayer

   RNNLayer
   BiRNNLayer

   ConvRNNCell
   BasicConvLSTMCell
   ConvLSTMLayer

   advanced_indexing_op
   retrieve_seq_length_op
   retrieve_seq_length_op2
   retrieve_seq_length_op3
   target_mask_op
   DynamicRNNLayer

   Seq2Seq
   PeekySeq2Seq
   AttentionSeq2Seq

   FlattenLayer
   ReshapeLayer
   TransposeLayer
   LambdaLayer

   ConcatLayer
   ElementwiseLayer

   ExpandDimsLayer
   TileLayer

   StackLayer
   UnStackLayer

   SlimNetsLayer

   BinaryDenseLayer
   BinaryConv2d
   TernaryDenseLayer
   TernaryConv2d
   DorefaDenseLayer
   DorefaConv2d
   SignLayer
   ScaleLayer

   PReluLayer

   MultiplexerLayer

   flatten_reshape
   clear_layers_name
   initialize_rnn_state
   list_remove_repeat
   merge_networks

名称与参数复用
---------------------------------

这些函数用以帮助您在不同的 graph 中复用相同的参数，以及如何通过一个名字来获取相应的参数列表。
更多关于 TensorFlow parameters sharing 请点击 `这里 <https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html>`_。

Get variables with name
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: get_variables_with_name

Get layers with name
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: get_layers_with_name

Enable layer name reuse
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: set_name_reuse

Print variables
^^^^^^^^^^^^^^^^^^
.. autofunction:: print_all_variables

Initialize variables
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: initialize_global_variables

Basic 层
----------------

.. autoclass:: Layer

输入层
----------

.. autoclass:: InputLayer
  :members:


One-hot 输入层
----------------
.. autoclass:: OneHotInputLayer

嵌入层+输入层
-----------------

训练Word2vec的层
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Word2vecEmbeddingInputlayer

嵌入层(输入)
^^^^^^^^^^^^^
.. autoclass:: EmbeddingInputlayer

平均化嵌入层(输入)
^^^^^^^^^^^^^^^^^^
.. autoclass:: AverageEmbeddingInputlayer

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

高斯噪声层
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GaussianNoiseLayer

Dropconnect + 全链接层
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DropconnectDenseLayer

卷积层 (Pro)
--------------------

1D 卷积层
^^^^^^^^^^

.. autoclass:: Conv1dLayer


2D 卷积层
^^^^^^^^^^

.. autoclass:: Conv2dLayer

2D 反卷积层
^^^^^^^^^^

.. autoclass:: DeConv2dLayer

3D 卷积层
^^^^^^^^^^

.. autoclass:: Conv3dLayer

3D 反卷积层
^^^^^^^^^^

.. autoclass:: DeConv3dLayer


2D 上采样层
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UpSampling2dLayer

2D 下采样层
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DownSampling2dLayer

2D 可变形卷积
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DeformableConv2dLayer

1D 多孔卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: AtrousConv1dLayer

2D多孔卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AtrousConv2dLayer


卷积层 (Simplified)
-----------------------------------

对于不擅长 TensorFlow 的用户，下面的简化的函数使用起来更简单。接下来我们将添加更多简化函数。

1D 卷积层
^^^^^^^^^
.. autoclass:: Conv1d

2D 卷积层
^^^^^^^^^
.. autoclass:: Conv2d

2D 反卷积层
^^^^^^^^^^^
.. autoclass:: DeConv2d

3D 反卷积层
^^^^^^^^^^^
.. autoclass:: DeConv3d

2D Depthwise 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DepthwiseConv2d

1D Depthwise Separable 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SeparableConv1d

2D Depthwise Separable 卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SeparableConv2d

2D 分组卷积层
^^^^^^^^^^^^^
.. autoclass:: GroupConv2d

Super-Resolution 层
-----------------------

1D 子像素卷积层
^^^^^^^^^^^^^^^^
.. autoclass:: SubpixelConv1d

2D 子像素卷积层
^^^^^^^^^^^^^^^^
.. autoclass:: SubpixelConv2d


空间变换网络
-----------------------

2D 仿射变换层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpatialTransformer2dAffineLayer

2D 仿射变换函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: transformer

批 2D 仿射变换函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: batch_transformer


池化和填充层
--------------------

填充层 (Pro)
^^^^^^^^^^^^^^
该填充层可以实现任意模式的填充。

.. autoclass:: PadLayer

池化层 (Pro)
^^^^^^^^^^^^
该池化层可以实现各种纬度（1D，2D，3D）以及各种池化方法（Mean，Max）。

.. autoclass:: PoolLayer

1D 补零填充层
^^^^^^^^^^^^^^^^^^^
.. autofunction:: ZeroPad1d

2D 补零填充层
^^^^^^^^^^^^^^^^^^^
.. autofunction:: ZeroPad2d

3D 补零填充层
^^^^^^^^^^^^^^^^^^^
.. autofunction:: ZeroPad3d

1D Max池化层
^^^^^^^^^^^^^
.. autoclass:: MaxPool1d

1D Mean池化层
^^^^^^^^^^^^^^
.. autoclass:: MeanPool1d

2D Max池化层
^^^^^^^^^^^^^
.. autoclass:: MaxPool2d

2D Mean池化层
^^^^^^^^^^^^^^
.. autoclass:: MeanPool2d

3D Max池化层
^^^^^^^^^^^^^
.. autoclass:: MaxPool3d

3D Mean池化层
^^^^^^^^^^^^^^
.. autoclass:: MeanPool3d

1D 全局Max池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool1d

1D 全局Mean池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool1d

2D 全局Max池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool2d

2D 全局Mean池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool2d

3D 全局Max池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool3d

3D 全局Mean池化层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool3d


规范化层
----------

Local Response Normalization 不包含任何参数，也没有复杂的设置。您也可以在 ``network.outputs`` 上使用 ``tf.nn.lrn()`` 来实现之。

Batch Normalization
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNormLayer

Local Response Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LocalResponseNormLayer

Instance Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InstanceNormLayer

Layer Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LayerNormLayer

物体检测
-------------------

ROI 层
^^^^^^^^^
.. autoclass:: ROIPoolingLayer



TimeDistributed 包装器
-------------------------
.. autoclass:: TimeDistributedLayer


定长递归层
----------------

所有递归层可实现任意 RNN 内核，只需输入不同的 cell 函数。

递归层
^^^^^^^^^^^^
.. autoclass:: RNNLayer

双向递归层
^^^^^^^^^^^^^
.. autoclass:: BiRNNLayer




卷积递归层
-------------------------------

Conv RNN Cell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ConvRNNCell

Basic Conv LSTM Cell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BasicConvLSTMCell

Conv LSTM layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ConvLSTMLayer




动态递归的高级 Ops 函数
-----------------------
这些函数通常在使用 Dynamic RNN layer 时使用，他们用以计算不同情况下的 sequence length，以及已知 sequence length 时对
输出进行索引。

输出索引
^^^^^^^^^^^^
用以得到 last output。

.. autofunction:: advanced_indexing_op

计算 Sequence length 方法1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: retrieve_seq_length_op

计算 Sequence length 方法2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: retrieve_seq_length_op2

计算 Sequence length 方法3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: retrieve_seq_length_op3

计算 Mask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: target_mask_op

动态递归层
------------
.. autoclass:: DynamicRNNLayer

动态递归层
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DynamicRNNLayer

动态双向递归层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BiDynamicRNNLayer


序列到序列
----------------------

Simple Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Seq2Seq

PeekySeq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: PeekySeq2Seq

AttentionSeq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AttentionSeq2Seq


形状修改层
----------------

Flatten层
^^^^^^^^^^^

.. autoclass:: FlattenLayer

Reshape层
^^^^^^^^^^^
.. autoclass:: ReshapeLayer

Transpose层
^^^^^^^^^^^^^
.. autoclass:: TransposeLayer


Lambda层
-----------
.. autoclass:: LambdaLayer

合并层
----------

Concat层
^^^^^^^^^^
.. autoclass:: ConcatLayer

Elementwise层
^^^^^^^^^^^^^^^^^
.. autoclass:: ElementwiseLayer

扩充层
---------

Expand 层
^^^^^^^^^^
.. autoclass:: ExpandDimsLayer

Tile 层
^^^^^^^^^^^^^^^^^
.. autoclass:: TileLayer



堆叠层
-------------

Stack 层
^^^^^^^^^^^^^^
.. autoclass:: StackLayer

Unstack 层
^^^^^^^^^^^^^^^
.. autofunction:: UnStackLayer



..
  Estimator 层
  ------------------
  .. autoclass:: EstimatorLayer

连接 TF-Slim
---------------

没错！TF-Slim 可以和 TensorLayer 无缝对接！所有 Google 预训练好的模型都可以直接使用！
模型请见 `Slim-model <https://github.com/tensorflow/models/tree/master/slim#Install>`_ 。

.. autoclass:: SlimNetsLayer

..
  连接 Keras
  ------------------

  没错！Keras 可以和 TensorLayer 无缝对接！ 参见 `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_ .

  .. autoclass:: KerasLayer


量化网络
------------------

先阅读这里
^^^^^^^^^^^^^^

这是一套搭建量化网络的试验版本API。
目前，我们依然用矩阵乘法而不是加减或bitcount运算来加速。
因此，这套API并不能加速，关于产品部署，目前您可以用TensorLayer训练模型，然后用自定义的C/C++实现的二值化计算（我们有可能会提供一套额外的专门运行二值化网络的框架，并支持可以从TensorLayer那读取模型）。

注意，作为试验版本，这套API有可能会被修改。

二值化全连接层
^^^^^^^^^^^^^^^^^
.. autoclass:: BinaryDenseLayer

二值化 2D 卷积层
^^^^^^^^^^^^^^^^^^
.. autoclass:: BinaryConv2d

Ternary 全连接层
^^^^^^^^^^^^^^^^^^
.. autoclass:: TernaryDenseLayer

Ternary 2D 卷积层
^^^^^^^^^^^^^^^^^^
.. autoclass:: TernaryConv2d

Dorefa 全连接层
^^^^^^^^^^^^^^^^^^
.. autoclass:: DorefaDenseLayer

Dorefa 2D 卷积层
^^^^^^^^^^^^^^^^^^
.. autoclass:: DorefaConv2d

Sign 层
^^^^^^^^^^^^^^
.. autoclass:: SignLayer

Scale 层
^^^^^^^^^^^^^^
.. autoclass:: ScaleLayer


带参数的激活函数
------------------

.. autoclass:: PReluLayer

流控制层
-----------

.. autoclass:: MultiplexerLayer

..
  包装器(Wrapper)
  ----------------

  嵌入+注意机制+Seq2seq
  ^^^^^^^^^^^^^^^^^^^^^^^

  .. autoclass:: EmbeddingAttentionSeq2seqWrapper
    :members:


辅助函数
----------------

Flatten tensor
^^^^^^^^^^^^^^^^^
.. autofunction:: flatten_reshape

永久删除现有 layer 名字
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: clear_layers_name

初始化 RNN state
^^^^^^^^^^^^^^^^^^^
.. autofunction:: initialize_rnn_state

去除列表中重复内容
^^^^^^^^^^^^^^^^^^
.. autofunction:: list_remove_repeat


合并网络属性
^^^^^^^^^^^^^^^^^^
.. autofunction:: merge_networks

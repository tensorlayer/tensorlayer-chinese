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

  cost = tl.cost.cross_entropy(y, y_)

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


了解Dense层
----------------

在创造自定义层之前，我们来看看全连接（Dense）层是如何实现的。
若不存在Weights矩阵和Biases向量时，它新建之，然后通过给定的激活函数计算出 ``outputs`` 。
在最后，作为一个有新参数的层，我们需要把新参数附加到 ``all_params`` 中。


.. code-block:: python

  class DenseLayer(Layer):
      """
      `DenseLayer` 是一个全连接层

      参数
      ----------
      layer : 一个 `Layer` 实例
          输入一个 `Layer` 类。
      n_units : 一个整数
          The number of units of the layer.
      act : 激活函数 （activation function）
          TensorFlow 激活函数。
      W_init : Weights初始化器（weights initializer）
          Weight matrix的初始化器。
      b_init : Biases初始化器（biases initializer）
          Bias vector 的初始化器，若为None，则无 bias。
      W_init_args : 一个字典（dictionary）
          Weights 使用 tf.get_variable 建立时，输入 tf.get_variable 的参数。
      b_init_args : 一个字典（dictionary）
          Biases 使用 tf.get_variable 建立时，输入 tf.get_variable 的参数。
      name : 字符串或 None
          该层的名字。
      """
      def __init__(
          self,
          layer = None,
          n_units = 100,
          act = tf.nn.relu,
          W_init = tf.truncated_normal_initializer(stddev=0.1),
          b_init = tf.constant_initializer(value=0.0),
          W_init_args = {},
          b_init_args = {},
          name ='dense_layer',
      ):
          self.inputs = layer.outputs
          if self.inputs.get_shape().ndims != 2:
              raise Exception("The input dimension must be rank 2")
          n_in = int(self.inputs._shape[-1])
          self.n_units = n_units
          print("  tensorlayer:Instantiate DenseLayer %s: %d, %s" % (self.name, self.n_units, act))
          with tf.variable_scope(name) as vs:
              W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args )
              if b_init:
                  b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args )
                  self.outputs = act(tf.matmul(self.inputs, W) + b)#, name=name)
              else:
                  self.outputs = act(tf.matmul(self.inputs, W))

          # Hint : list(), dict() is pass by value (shallow).
          self.all_layers = list(layer.all_layers)
          self.all_params = list(layer.all_params)
          self.all_drop = dict(layer.all_drop)
          self.all_layers.extend( [self.outputs] )
          if b_init:
             self.all_params.extend( [W, b] )
          else:
             self.all_params.extend( [W] )

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
          Layer.__init__(self, name=name)
          self.inputs = layer.outputs
          self.outputs = self.inputs * 2

          self.all_layers = list(layer.all_layers)
          self.all_params = list(layer.all_params)
          self.all_drop = dict(layer.all_drop)
          self.all_layers.extend( [self.outputs] )



修改预训练行为
-----------------------

逐层贪婪预训练方法(Greedy layer-wise pretrain)是深度神经网络的初始化非常重要的一种方法，
不过对不同的网络结构和应用，往往有不同的预训练的方法。



For example, the pre-train process of `Vanilla Sparse Autoencoder <http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity>`_
can be implemented by using KL divergence as the following code,
but for `Deep Rectifier Network <http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf>`_,
the sparsity can be implemented by using the L1 regularization of activation output.


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
   PoolLayer
   PadLayer
   UpSampling2dLayer
   DownSampling2dLayer
   AtrousConv2dLayer
   SeparableConv2dLayer

   Conv2d
   DeConv2d
   
   MaxPool1d
   MeanPool1d
   MaxPool2d
   MeanPool2d
   MaxPool3d
   MeanPool3d

   BatchNormLayer
   LocalResponseNormLayer

   TimeDistributedLayer

   RNNLayer
   BiRNNLayer
   advanced_indexing_op
   retrieve_seq_length_op
   retrieve_seq_length_op2
   DynamicRNNLayer

   Seq2Seq
   PeekySeq2Seq
   AttentionSeq2Seq

   FlattenLayer
   ReshapeLayer
   LambdaLayer

   ConcatLayer
   ElementwiseLayer

   ExpandDimsLayer
   TileLayer

   SlimNetsLayer
   KerasLayer

   PReluLayer

   MultiplexerLayer

   EmbeddingAttentionSeq2seqWrapper

   flatten_reshape
   clear_layers_name
   initialize_rnn_state
   list_remove_repeat

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

卷积层(Pro)
--------------------

1D卷积层
^^^^^^^^^^

.. autoclass:: Conv1dLayer


2D卷积层
^^^^^^^^^^

.. autoclass:: Conv2dLayer

2D反卷积层
^^^^^^^^^^

.. autoclass:: DeConv2dLayer

3D卷积层
^^^^^^^^^^

.. autoclass:: Conv3dLayer

3D反卷积层
^^^^^^^^^^

.. autoclass:: DeConv3dLayer


2D上采样层
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UpSampling2dLayer

2D下采样层
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DownSampling2dLayer

2D多孔卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AtrousConv2dLayer

2D Separable卷积层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SeparableConv2dLayer

卷积层(Simplified)
-----------------------------------

对于不擅长 TensorFlow 的用户，下面的简化的函数使用起来更简单。接下来我们将添加更多简化函数。

2D卷积层
^^^^^^^^^

.. autofunction:: Conv2d

2D反卷积层
^^^^^^^^^^^

.. autofunction:: DeConv2d

1D Max池化层
^^^^^^^^^^^^^

.. autofunction:: MaxPool1d

1D Mean池化层
^^^^^^^^^^^^^^

.. autofunction:: MeanPool1d

2D Max池化层
^^^^^^^^^^^^^

.. autofunction:: MaxPool2d

2D Mean池化层
^^^^^^^^^^^^^^

.. autofunction:: MeanPool2d

3D Max池化层
^^^^^^^^^^^^^

.. autofunction:: MaxPool3d

3D Mean池化层
^^^^^^^^^^^^^^

.. autofunction:: MeanPool3d

池化层
--------------------

该池化层可以实现各种纬度（1D，2D，3D）以及各种池化方法（Mean，Max）。

.. autoclass:: PoolLayer


填充层
--------------------
该填充层可以实现任意模式的填充。

.. autoclass:: PadLayer


规范化层
----------

Local Response Normalization 不包含任何参数，也没有复杂的设置。您也可以在 ``network.outputs`` 上使用 ``tf.nn.lrn()`` 来实现之。

Batch Normalization
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNormLayer

Local Response Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LocalResponseNormLayer

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


连接 TF-Slim
---------------

没错！TF-Slim 可以和 TensorLayer 无缝对接！所有 Google 预训练好的模型都可以直接使用！
模型请见 `Slim-model <https://github.com/tensorflow/models/tree/master/slim#Install>`_ 。

.. autoclass:: SlimNetsLayer

连接 Keras
------------------

没错！Keras 可以和 TensorLayer 无缝对接！ 参见 `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_ .

.. autoclass:: KerasLayer



带参数的激活函数
------------------

.. autoclass:: PReluLayer

流控制层
-----------

.. autoclass:: MultiplexerLayer

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

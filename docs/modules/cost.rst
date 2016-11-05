API - 损失函数
=======================

为了尽可能地保持TensorLayer的简洁性，我们最小化损失函数的数量。
因此我们鼓励直接使用TensorFlow官方的函数，比如你可以通过
``tf.contrib.layers.l1_regularizer``, ``tf.contrib.layers.l2_regularizer`` and
``tf.contrib.layers.sum_regularizer`` 来实现L1, L2 和 sum 规则化， 参考 `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_。


自定义损失函数
-----------------------

TensorLayer提供一个简单的方法来创建您自己的损失函数。
下面以多层神经网络(MLP)为例：

.. code-block:: python

  network = tl.InputLayer(x, name='input_layer')
  network = tl.DropoutLayer(network, keep=0.8, name='drop1')
  network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
  network = tl.DropoutLayer(network, keep=0.5, name='drop2')
  network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
  network = tl.DropoutLayer(network, keep=0.5, name='drop3')
  network = tl.DenseLayer(network, n_units=10, act = tl.activation.identity, name='output_layer')

那么其模型参数为 ``[W1, b1, W2, b2, W_out, b_out]``，
这时，你可以像下面的例子那样实现对前两个weights矩阵的L2规则化。

.. code-block:: python

  cost = tl.cost.cross_entropy(y, y_)
  cost = cost + tf.contrib.layers.l2_regularizer(0.001)(network.all_params[0]) + tf.contrib.layers.l2_regularizer(0.001)(network.all_params[2])

此外，TensorLayer 提供了通过给定名称，很方便地获取参数列表的方法，所以您可以如下对某些参数执行L2规则化。

.. code-block:: python

  l2 = 0
  for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=False):#[-3:]:
      l2 += tf.contrib.layers.l2_regularizer(1e-4)(w)
  cost = tl.cost.cross_entropy(y, y_) + l2



权值的正则化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在初始化变量之后，网络参数的信息可以使用 ``network.print.params()`` 来获得。

.. code-block:: python

  sess.run(tf.initialize_all_variables())
  network.print_params()

.. code-block:: text

  param 0: (784, 800) (mean: -0.000000, median: 0.000004 std: 0.035524)
  param 1: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  param 2: (800, 800) (mean: 0.000029, median: 0.000031 std: 0.035378)
  param 3: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  param 4: (800, 10) (mean: 0.000673, median: 0.000763 std: 0.049373)
  param 5: (10,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  num of params: 1276810

网络的输出是 ``network.outputs`` ，那么交叉熵的可以被如下定义。
另外，要正则化权重， ``network.all_params`` 要包含网络的所有参数。
在这种情况下根据 ``network.print_params()`` 所展示的参数 0,1,...,5的值, ``network.all_params =  [W1, b1, W2, b2, Wout, bout]``
然后对W1和W2的最大范数正则化可以按如下进行：

.. code-block:: python

  y = network.outputs
  # Alternatively, you can use tl.cost.cross_entropy(y, y_) instead.
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
  cost = cross_entropy
  cost = cost + tl.cost.maxnorm_regularizer(1.0)(network.all_params[0]) +
            tl.cost.maxnorm_regularizer(1.0)(network.all_params[2])

另外，所有的TensorFlow的正则化函数，像 ``tf.contrib.layers.l2_regularizer`` 在TensorLayer中也能使用。

激活输出(Activation outputs)的规则化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

实例方法 ``network.print_layers()`` 整齐地打印不同层的所有输出。
为了实现对激活输出的正则化，您可以使用 ``network.all_layers`` ，它包含了不同层的所有输出。
如果您想对第一层隐藏层的激活输出使用L1惩罚，仅仅需要添加
``tf.contrib.layers.l2_regularizer(lambda_l1)(network.all_layers[1])`` 到成本函数中。

.. code-block:: python

  network.print_layers()

.. code-block:: text

  layer 0: Tensor("dropout/mul_1:0", shape=(?, 784), dtype=float32)
  layer 1: Tensor("Relu:0", shape=(?, 800), dtype=float32)
  layer 2: Tensor("dropout_1/mul_1:0", shape=(?, 800), dtype=float32)
  layer 3: Tensor("Relu_1:0", shape=(?, 800), dtype=float32)
  layer 4: Tensor("dropout_2/mul_1:0", shape=(?, 800), dtype=float32)
  layer 5: Tensor("add_2:0", shape=(?, 10), dtype=float32)





.. automodule:: tensorlayer.cost

.. autosummary::

   cross_entropy
   binary_cross_entropy
   mean_squared_error
   dice_coe
   iou_coe
   cross_entropy_seq
   li_regularizer
   lo_regularizer
   maxnorm_regularizer
   maxnorm_o_regularizer
   maxnorm_i_regularizer

损失函数
----------------

.. autofunction:: cross_entropy
.. autofunction:: binary_cross_entropy
.. autofunction:: mean_squared_error
.. autofunction:: dice_coe
.. autofunction:: iou_coe
.. autofunction:: cross_entropy_seq


规则化函数
--------------------------

.. autofunction:: li_regularizer
.. autofunction:: lo_regularizer
.. autofunction:: maxnorm_regularizer
.. autofunction:: maxnorm_o_regularizer
.. autofunction:: maxnorm_i_regularizer

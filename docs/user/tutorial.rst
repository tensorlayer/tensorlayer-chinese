.._tutorial:

=======
教程
=======

对于深度学习，这篇教程会引导您使用MNIST数据集构建一个手写数字的分类器，
这可以说是神经网络的 "Hello World" 。
对于强化学习，我们将让计算机从屏幕输入来学习打乒乓球。
对于自然语言处理。我们从词嵌套(word embedding)开始，然后描述语言模型和机器翻译。

.. note::
    对于专家们来说，阅读 ``InputLayer`` 和 ``DenseLayer`` 文档，您将会理解TensorLayer是如何工作的。
    然后，我们建议您直接去教程的源代码。

在我们开始之前
==================

本教程假定您在神经网络和TensorFlow(TensorLayer在它的基础上构建的)方面具有一定的基础。
您可以尝试从 `Deeplearning Tutorial`_ 同时进行学习。

对于人工神经网络更系统的介绍，我们推荐Andrej Karpathy等人所著的 `Convolutional Neural Networks for Visual Recognition`_
和Michael Nielsen `Neural Networks and Deep Learning`_ 。

要了解TensorFlow的更多内容，请阅读 `TensorFlow tutorial`_ 。
您不需要会它的全部，只要知道TensorFlow是如何工作的，就能够使用TensorLayer。
如果您是TensorFlow的新手，建议你阅读整个教程。

运行MNIMST样例
=====================

.. _fig_0601:

.. image:: my_figs/mnist.jpeg
  :scale: 100 %
  :align: center

在本教程的第一部分，我们仅仅运行TensorLayer内置的MNIST样例。
MNIST数据集包含了60000个28x28像素的手写数字图片，它通常用于训练各种图片识别系统。

我们假设您已经按照 :ref:`installation` 安装过TensorLayer。
如果您还没有，请复制一个TensorLayer的source目录到终端中进入该文件夹，
然后运行 ``tutorial_mnist.py`` 示例脚本：

.. code-block:: bash

  python tutorial_mnist.py

If everything is set up correctly, you will get an output like the following:

.. code-block:: text

  tensorlayer: GPU MEM Fraction 0.300000
  Downloading train-images-idx3-ubyte.gz
  Downloading train-labels-idx1-ubyte.gz
  Downloading t10k-images-idx3-ubyte.gz
  Downloading t10k-labels-idx1-ubyte.gz

  X_train.shape (50000, 784)
  y_train.shape (50000,)
  X_val.shape (10000, 784)
  y_val.shape (10000,)
  X_test.shape (10000, 784)
  y_test.shape (10000,)
  X float32   y int64

  tensorlayer:Instantiate InputLayer input_layer (?, 784)
  tensorlayer:Instantiate DropoutLayer drop1: keep: 0.800000
  tensorlayer:Instantiate DenseLayer relu1: 800, <function relu at 0x11281cb70>
  tensorlayer:Instantiate DropoutLayer drop2: keep: 0.500000
  tensorlayer:Instantiate DenseLayer relu2: 800, <function relu at 0x11281cb70>
  tensorlayer:Instantiate DropoutLayer drop3: keep: 0.500000
  tensorlayer:Instantiate DenseLayer output_layer: 10, <function identity at 0x115e099d8>

  param 0: (784, 800) (mean: -0.000053, median: -0.000043 std: 0.035558)
  param 1: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  param 2: (800, 800) (mean: 0.000008, median: 0.000041 std: 0.035371)
  param 3: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  param 4: (800, 10) (mean: 0.000469, median: 0.000432 std: 0.049895)
  param 5: (10,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  num of params: 1276810

  layer 0: Tensor("dropout/mul_1:0", shape=(?, 784), dtype=float32)
  layer 1: Tensor("Relu:0", shape=(?, 800), dtype=float32)
  layer 2: Tensor("dropout_1/mul_1:0", shape=(?, 800), dtype=float32)
  layer 3: Tensor("Relu_1:0", shape=(?, 800), dtype=float32)
  layer 4: Tensor("dropout_2/mul_1:0", shape=(?, 800), dtype=float32)
  layer 5: Tensor("add_2:0", shape=(?, 10), dtype=float32)

  learning_rate: 0.000100
  batch_size: 128

  Epoch 1 of 500 took 0.342539s
    train loss: 0.330111
    val loss: 0.298098
    val acc: 0.910700
  Epoch 10 of 500 took 0.356471s
    train loss: 0.085225
    val loss: 0.097082
    val acc: 0.971700
  Epoch 20 of 500 took 0.352137s
    train loss: 0.040741
    val loss: 0.070149
    val acc: 0.978600
  Epoch 30 of 500 took 0.350814s
    train loss: 0.022995
    val loss: 0.060471
    val acc: 0.982800
  Epoch 40 of 500 took 0.350996s
    train loss: 0.013713
    val loss: 0.055777
    val acc: 0.983700
  ...

示例脚本允许您从 ``if__name__=='__main__':`` 中选择不同的模型进行尝试，包括多层神经网络(Multi-Layer Perceptron)，
Dropout, Dropconnect, Stacked Denoising Autoencoder and 卷积神经网络.

.. code-block:: python

  main_test_layers(model='relu')
  main_test_denoise_AE(model='relu')
  main_test_stacked_denoise_AE(model='relu')
  main_test_cnn_layer()



理解MNIST样例
=====================

现在，让我们研究它是怎么做到的！跟上脚步，打开源代码。

序言
-----------

您第一件可能注意到的是除了TensorLayer之外，我们还导入了numpy和tensorflow：

.. code-block:: python

  import tensorflow as tf
  import tensorlayer as tl
  from tensorlayer.layers import set_keep
  import numpy as np
  import time


正如我们所知，TensorLayer是建立在TensorFlow上的，目的是为某些任务的提供充分的帮助而不是取代它。
您总会联用TensorLayer和一些普通的TensorFlw代码。当使用降噪自编码器(Denoising Autoencoder)时，
``set_keep`` 常用来访问保持概率(keep probabilities)的占位符。


载入数据
-------------

第一块代码定义了 ``load_mnist_dataset()`` 函数。
其目的时下载MNIST数据集(如果它还没有被下载的话)并且返回标准numpy数列(numpy array)的格式。
这完全没有涉及TensorLayer，所以出于这个教程的目的，我们可以把它看作：

.. code-block:: python

  X_train, y_train, X_val, y_val, X_test, y_test = \
                    tl.files.load_mnist_dataset(shape=(-1,784))


``X_train.shape`` 是 ``(50000,784)`` ，翻译过来就是50000张图片并且每张图片有784个像素点。
``Y_train.shape`` 是 ``(50000,)`` ，它是一个和 ``X_train`` 长度一样的向量，给出了每幅图的标签
——即0到9之间这张图片显示的数字(写这个数字的人注释的)

对于卷积神经网络的例子，MNIST可以按下列的4D版本载入：

.. code-block:: python

  X_train, y_train, X_val, y_val, X_test, y_test = \
              tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

``X_train.shape`` 是 ``(50000,28,28,1)`` ，这代表了50000张图片，每张图片使用一个信道，28行，28列。
信道为1是因为它是灰度图像，每个像素只能有一个值。

建立模型
----------------

TensorLayer只需要几步就可以完成这个工作。TensorLayer允许您通过创建，堆叠或者合并图层来定义任意结构的神经网络。
由于每一层都知道它在一个网络中的直接输入层和（多个）输出接收层，就像xxxxx一样。
通常这是我们唯一传递给其他代码的内容。

如上所述， ``tutorial_mnist.py`` 支持四类模型，并且我们很容易通过改变同一接口的函数来实现模型。
首先，我们仔细地按照步骤说明，定义一个生成一种固定结构的多层次感知器的函数。
然后，我们将实现一个去噪自编码器(Denosing Autoencoding)。
再之后，我们要将所有地去噪自编码器堆叠起来并且监督式地对他们进行微调。
最后，我们将展示如何创建一个卷积神经网络(Convolutional Neural Network)。

多层次感知器(Multi-Layer Perceptron)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

第一个脚本 ``main_test_layers()`` ,创建了一个具有两个隐藏层，每层800个单元的多层次感知器并且具有10个单元的SOFTMAX输出层紧随其后。
它对输入数据采用20%的退出率(dropout)并且对隐藏层应用50%的退出率(dropout)。

为了喂数据给这个网络，TensorFlow占位符需要按如下定义。
在这里 ``None`` 是指在编译之后，网络将接受任意批规模(batchsize)的数据
``x`` 是用来存放 ``X_train`` 数据的并且 ``y_`` 是用来存放 ``y_train`` 数据的。
如果实现知道批规模，那就不需要这种灵活性了。您可以在这里给出批规模，特别是对于卷积层，这样可以让TensorFlow得到一些优化。

.. code-block:: python

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

在TensorLayer中每个神经网络的基础是一个 :class:`InputLayer <tensorlayer.layers.InputLayer>` 实例。它代表了将要喂给网络的输入数据。
值得注意的是 ``InputLayer`` 并不依赖任何特定的数据的。

.. code-block:: python

    network = tl.layers.InputLayer(x, name='input_layer')

在添加第一层隐藏层之前，我们要对输入数据应用20%的退出率(dropout)。
这里我们是通过一个 :class:`DropoutLayer<tensorlayer.layers.DropoutLayer>` 的实例来实现的。

.. code-block:: python

    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')

注意！构造函数的第一个参数是输入层，第二个参数是激活值的保持概率(keeping probability for the activation value)
现在我们要继续构造第一个800个单位的全连接的隐藏层。
尤其是当要堆叠一个 :class:`DenseLayer <tensorlayer.layers.DenseLayer>` 时要注意这个。

.. code-block:: python

    network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')

同样，构造函数的顶一个参数以为这我们正在 ``network`` 之上堆叠 ``network`` 。
``n_units`` 仅仅时给出了全连接层的单位数。
``act`` 给出了一个激活函数，这里是 :mod:`tensorflow.nn` 和  `tensorlayer.activation` 中所定义的几个函数。
我们在这里选择了整流器(rectifier)，所以我们将得到ReLUs
我们现在添加50%的退出率，对于另一个800单位的稠密层(dense layer)，我们也添加50%的退出率：

.. code-block:: python

    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')

最后，我们加入 ``n_units`` 等于分类个数的全连接的输出层。

.. code-block:: python

    network = tl.layers.DenseLayer(network,
                                  n_units=10,
                                  act = tl.activation.identity,
                                  name='output_layer')

如上所述，每层被链接到它的输入层,所以我们只需要在TensorLayer中将输出层接入一个网络：

.. code-block:: python

    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

在这里，``network.outputs`` 是网络的10个特征的输出(按照一个热格式(hot format))。
``y_op`` 是代表类索引的整数输出， ``cost`` 是目标和预测标签的交叉熵。

去噪自编码器(Denoising Autoencoder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

自编码器是一种能够提取具有代表性特征的无监督学习模型，
它已经广泛使用于数据生成模式的学习与逐层贪婪的预训练(Greedy layer-wise pre-train)。

脚本 ``main_test_denoise_AE()`` 实现了有50%的腐蚀率(corrosion rate)的去噪自编码器。
这个自编码器可以按如下方式定义，这里一个 ``DenseLayer`` 代表一个 自编码器：

.. code-block:: python

    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='denoising1')
    network = tl.layers.DenseLayer(network, n_units=200, act=tf.nn.sigmoid, name='sigmoid1')
    recon_layer1 = tl.layers.ReconLayer(network,
                                        x_recon=x,
                                        n_units=784,
                                        act=tf.nn.sigmoid,
                                        name='recon_layer1')

训练 ``DenseLayer`` ，只需要运行 ``ReconLayer.Pretrain()`` 即可。
如果要使用去噪自编码器，腐蚀层(corrosion layer)(``DropoutLayer``)的名字需要按后面说的指定。
如果要保存特征图像，设置 ``save`` 为 True 。
灯具不同的架构和应用这里可以设置许多预训练的度量(metric)

对于 sigmoid型激活函数来说，自编码器可以用KL散度来实现。
而对于 整流器(rectifier)来说，对激活函数输出的L1正则化能使得输出投影到稀疏空间中。
所以 ``ReconLayer`` 的默认行为只对整流激活函数提供sigmoid型激活函数，L1正则化激活输出和均方差的KLD和交叉熵
我们建立您修改 ``ReconLayer`` 来实现自己的预训练度量。

.. code-block:: python

    recon_layer1.pretrain(sess,
                          x=x,
                          X_train=X_train,
                          X_val=X_val,
                          denoise_name='denoising1',
                          n_epoch=200,
                          batch_size=128,
                          print_freq=10,
                          save=True,
                          save_name='w1pre_')

此外，脚本 ``main_test_stacked_denoise_AE()`` 展示了如何将多个自编码器堆叠到一个网络，然后进行微调。

卷积神经网络(Convolutional Neural Network)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最后，``main_test_cnn_layer()`` 脚本创建了两个CNN 层和最大汇流(max pooling stages)，一个全连接的隐藏层和一个全连接的输出层。

首先，我们添加一个 :class:`Conv2dLayer<tensorlayer.layers.Conv2dLayer>` ，
它顶部有32个5x5的滤波器，紧接着在两个2个向量的同尺寸的最大汇流。


.. code-block:: python

    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                            act = tf.nn.relu,
                            shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name ='cnn_layer1')     # output: (?, 28, 28, 32)
    network = tl.layers.PoolLayer(network,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool = tf.nn.max_pool,
                            name ='pool_layer1',)   # output: (?, 14, 14, 32)
    network = tl.layers.Conv2dLayer(network,
                            act = tf.nn.relu,
                            shape = [5, 5, 32, 64], # 64 features for each 5x5 patch
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name ='cnn_layer2')     # output: (?, 14, 14, 64)
    network = tl.layers.PoolLayer(network,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool = tf.nn.max_pool,
                            name ='pool_layer2',)   # output: (?, 7, 7, 64)
    network = tl.layers.FlattenLayer(network, name='flatten_layer')
                                                    # output: (?, 3136)
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
                                                    # output: (?, 3136)
    network = tl.layers.DenseLayer(network, n_units=256, act = tf.nn.relu, name='relu1')
                                                    # output: (?, 256)
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
                                                    # output: (?, 256)
    network = tl.layers.DenseLayer(network, n_units=10, act = tl.identity, name='output_layer')
                                                    # output: (?, 10)

.. note::
    对于专家们来说， ``Conv2dLayer`` 将使用 ``tensorflow.nn.conv2d`` ,TensorFlow默认的卷积方式来创建一个卷积层。

训练模型
----------------

在 ``tutorial_mnist.py`` 脚本的其余部分对在MNIST数据上只使用交叉熵的循环训练进行了设置并且运行

数据集迭代
^^^^^^^^^^^^^

一个在给定的项目数的最小批规模下的输入特征及其对应的标签的两Numpy数列依次同步的迭代函数。
更多的迭代函数可以在 ``tensorlayer.iterate`` 中找到。

.. code-block:: python

    tl.iterate.minibatches(inputs, targets, batchsize, shuffle=False)

损失和更新公式

我们继续创建一个在训练中被最小化的损失表达式：

.. code-block:: python

    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

采用 ``main_test_layers()`` 更多的成本或者正则化方法可以被应用在这里。
例如:要在权重矩阵中应用最大模(max-norm)方法，你可以添加下列行：

.. code-block:: python

    cost = cost + tl.cost.maxnorm_regularizer(1.0)(network.all_params[0]) +
                  tl.cost.maxnorm_regularizer(1.0)(network.all_params[2])

根据您要解决的问题，您会需要不同的损失函数，更多的损失函数请见： `tensorlayer.cost`

有了模型和定义的损失函数之后，我们就可以创建用于训练网络的更新公式。
TensorLayer不自身提供优化，我们使用TensorFlow的优化。

.. code-block:: python

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

为了训练网络，我们要提供数据和保持概率给 ``feed_dict``。

.. code-block:: python

    feed_dict = {x: X_train_a, y_: y_train_a}
    feed_dict.update( network.all_drop )
    sess.run(train_op, feed_dict=feed_dict)

同时为了进行验证和测试，我们采用略有不同的方法。
所有的退出，退连(dropconnect)，腐蚀层(corrosion layers)都要被禁用。

.. code-block:: python

    dp_dict = tl.utils.dict_to_one( network.all_drop )
    feed_dict = {x: X_test_a, y_: y_test_a}
    feed_dict.update(dp_dict)
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)

作为一个额外的监测量，我们创建一个分类准确度的公式：

.. code-block:: python

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

下一步？
^^^^^^^^^^^^^^

在 ``tutorial_cifar10.py`` 中我们也有更高级的图像分类的样例。
请阅读代码及注释，弄清楚如何产生更多的训练数据和什么是局部响应正则化。
在这之后，尝试实现 `残差网络(Residual Network) <http://doi.org/10.3389/fpsyg.2013.00124>`_
*提示：您会用到Layer.outputs。*


运行乒乓实例
====================

在本教程的第二部分，我们将运行一个深度强化学习的实例，它在Karpathy的 `Deep Reinforcement Learning:Pong from Pixels <http://karpathy.github.io/2016/05/31/rl/>`_ 有介绍。

.. code-block:: bash

  python tutorial_atari_pong.py

在运行教程代码之前 你需要安装 `OpenAI gym environment <https://gym.openai.com/docs>`_ ,它是强化学习的一个标杆。
如果一切设置正确，您将得到一个类似以下的输出：

.. code-block:: text

  [2016-07-12 09:31:59,760] Making new env: Pong-v0
    tensorlayer:Instantiate InputLayer input_layer (?, 6400)
    tensorlayer:Instantiate DenseLayer relu1: 200, <function relu at 0x1119471e0>
    tensorlayer:Instantiate DenseLayer output_layer: 3, <function identity at 0x114bd39d8>
    param 0: (6400, 200) (mean: -0.000009, median: -0.000018 std: 0.017393)
    param 1: (200,) (mean: 0.000000, median: 0.000000 std: 0.000000)
    param 2: (200, 3) (mean: 0.002239, median: 0.003122 std: 0.096611)
    param 3: (3,) (mean: 0.000000, median: 0.000000 std: 0.000000)
    num of params: 1280803
    layer 0: Tensor("Relu:0", shape=(?, 200), dtype=float32)
    layer 1: Tensor("add_1:0", shape=(?, 3), dtype=float32)
  episode 0: game 0 took 0.17381s, reward: -1.000000
  episode 0: game 1 took 0.12629s, reward: 1.000000  !!!!!!!!
  episode 0: game 2 took 0.17082s, reward: -1.000000
  episode 0: game 3 took 0.08944s, reward: -1.000000
  episode 0: game 4 took 0.09446s, reward: -1.000000
  episode 0: game 5 took 0.09440s, reward: -1.000000
  episode 0: game 6 took 0.32798s, reward: -1.000000
  episode 0: game 7 took 0.74437s, reward: -1.000000
  episode 0: game 8 took 0.43013s, reward: -1.000000
  episode 0: game 9 took 0.42496s, reward: -1.000000
  episode 0: game 10 took 0.37128s, reward: -1.000000
  episode 0: game 11 took 0.08979s, reward: -1.000000
  episode 0: game 12 took 0.09138s, reward: -1.000000
  episode 0: game 13 took 0.09142s, reward: -1.000000
  episode 0: game 14 took 0.09639s, reward: -1.000000
  episode 0: game 15 took 0.09852s, reward: -1.000000
  episode 0: game 16 took 0.09984s, reward: -1.000000
  episode 0: game 17 took 0.09575s, reward: -1.000000
  episode 0: game 18 took 0.09416s, reward: -1.000000
  episode 0: game 19 took 0.08674s, reward: -1.000000
  episode 0: game 20 took 0.09628s, reward: -1.000000
  resetting env. episode reward total was -20.000000. running mean: -20.000000
  episode 1: game 0 took 0.09910s, reward: -1.000000
  episode 1: game 1 took 0.17056s, reward: -1.000000
  episode 1: game 2 took 0.09306s, reward: -1.000000
  episode 1: game 3 took 0.09556s, reward: -1.000000
  episode 1: game 4 took 0.12520s, reward: 1.000000  !!!!!!!!
  episode 1: game 5 took 0.17348s, reward: -1.000000
  episode 1: game 6 took 0.09415s, reward: -1.000000

这个例子让电脑从屏幕输入来学习如何像人类一样打乒乓球。
在经过15000个序列的训练之后，计算机就可以赢得20%的比赛。
在20000个序列的训练之后，计算机可以赢得35%的比赛，
我们可以看到计算机学的越来越快，这是因为它有更多的胜利的数据来进行训练。
如果您用30000个序列来训练它，那么它会一直赢。

.. code-block:: python

  render = False
  resume = False

如果您想显示游戏的环境，那就设置 `render` 为 `True` 。
当您再次运行该代码，您可以设置 `resume` 为 `True`,那么代码将加载现有的模型并且会基于它进行训练。

.. _fig_0601:

.. image:: my_figs/pong_game.jpeg
    :scale: 30 %
    :align: center

理解强化学习
===================

乒乓球
-------------

要理解强化学习，我们要让电脑学习如何从初始的屏幕输入打乒乓球。
在我们开始之前，我们强烈建议您去浏览一个著名的博客叫做 `Deep Reinforcement Learning:pong from Pixels <http://karpathy.github.io/2016/05/31/rl/>`_ ,
这是使用python numpy库和OpenAI gym environment=来实现的一个深度强化学习的最简实现。


.. code-block:: bash

  python tutorial_atari_pong.py

策略网络(Policy Network)
---------------------------

在深度强化学习中，Policy Network 等同于 深度神经网络。
它是我们的选手(或者说“代理人(agent)”），它的输出行为告诉我们应该做什么(向上移动或向下移动)：
在Karpathy的代码中，他值定理了2个动作，向上移动和向下移动，并且仅使用单个simgoid输出：
为了使我们的教程更具有普遍性，我们使用3个SOFTMAX输出来定义向上移动，向下移动和停止(什么都不做)3个动作。

.. code-block:: python

    # observation for training
    states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])

    network = tl.layers.InputLayer(states_batch_pl, name='input_layer')
    network = tl.layers.DenseLayer(network, n_units=H,
                                    act = tf.nn.relu, name='relu1')
    network = tl.layers.DenseLayer(network, n_units=3,
                            act = tl.activation.identity, name='output_layer')
    probs = network.outputs
    sampling_prob = tf.nn.softmax(probs)

然后我们的代理人就一直打乒乓球。它计算不同动作的概率，
并且之后会从这个均匀的分布中选取样本(动作)。
因为动作被1,2和3代表，但是softmax输出应该从0开始，所以我们从-1计算这个标签的价值。

.. code-block:: python

    prob = sess.run(
        sampling_prob,
        feed_dict={states_batch_pl: x}
    )
    # action. 1: STOP  2: UP  3: DOWN
    action = np.random.choice([1,2,3], p=prob.flatten())
    ...
    ys.append(action - 1)

策略逼近(Policy Gradient)
---------------------------

策略梯度下降法是一个end-to-end的算法，它直接学习从状态映射到动作的策略函数。
一个近似最优的策略可以通过最大化预期的奖励来直接学习。
策略函数的参数(例如，在乒乓球示例终使用的策略网络的参数)在预期奖励的近似值的引导下能够被训练和学习。
换句话说，我们可以通过过更新它的参数来逐步调整策略函数，这样它能从给定的状态做出一系列行为来获得更高的奖励。

策略迭代的一个替代算法就是深度Q-learning(DQN)。
他是基于Q-learning,学习一个映射状态和动作到一些值的价值函数的算法(叫Q函数)。
DQN采用了一个深度神经网络来作为Q函数的逼近来代表Q函数。
训练是通过最小化时序差分(temporal-difference)误差来实现。
一个名为“再体验(experience replay)”的神经生物学的启发式机制通常和DQN一起被使用来帮助提高非线性函数的逼近的稳定性

您可以阅读以下文档，来得到对强化学习更好的理解：



数据集迭代
^^^^^^^^^^^^^^

在强化学习中，我们考虑最终的决策来作为一个序列。在乒乓球游戏中，一个序列是几十场比赛，因为比赛对于其中一方，总有一个要达到21分。
然后批规模是多少支我们人为可以更新模型的序列。
在本教程中，我们在每批规模为10序列使用RMSProp训练一个具有200个单元的隐藏层的2层策略网络

损失和更新公式
^^^^^^^^^^^^^^^^^^^

接着我们创建一个在训练中被最小化的损失公式：

.. code-block:: python

    actions_batch_pl = tf.placeholder(tf.int32, shape=[None])
    discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])
    loss = tl.rein.cross_entropy_reward_loss(probs, actions_batch_pl,
                                                  discount_rewards_batch_pl)
    ...
    ...
    sess.run(
        train_op,
        feed_dict={
            states_batch_pl: epx,
            actions_batch_pl: epy,
            discount_rewards_batch_pl: disR
        }
    )

一批次的损失和一个批次内的策略网络的所有输出，所有的我们做出的动作和相应的被打折的奖励有关
我们首先通过累加被打折的奖励和实际输出和真实动作的交叉熵计算每一个动作的损失。
最后的损失是所有动作的损失的和。

下一步?
----------------

上述教程展示了您如何去建立自己的代理人，end-to-end。
虽然它有很合理的品质，但它的默认参数不会给你最好的代理人模型。
这有一些您可以优化的内容。

首先，与传统的MLP模型不同，比起  `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_ 更好的是我们可以使用CNNs来采集屏幕信息

另外这个模型默认参数没有调整，您可以更改学习率，衰退率，或者用不同的方式来初始化您的模型的权重。

最后，您可以尝试不同任务(游戏)的模型。

运行 Word2Vec实例：
====================

在教程的这一部分，我们训练一个词的矩阵，其中每个词可以通过唯一的行向量来矩阵表示。
在结束时，同样的话将会有类似的向量。
然后就像我们把词在一个2为平面熵画出来一样，那些相似的词最终会彼此聚集在一起。

.. code-block:: bash

  python tutorial_word2vec_basic.py

如果一切设置正确，您最后会得到一个输出。

.. _fig_0601:

.. image:: my_figs/tsne.png
  :scale: 100 %
  :align: center

理解词嵌套(word embedding)
=================================

词嵌套
----------------

董豪强烈建立您阅读Colah的博客 `Word Representations`_ 来理解为什么我们要使用向量来作为代表以及要如何计算这个向量。

训练一个嵌套矩阵

.. code-block:: python

  # train_inputs is a row vector, a input is an integer id of single word.
  # train_labels is a column vector, a label is an integer id of single word.
  # valid_dataset is a column vector, a valid set is an integer id of single word.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Look up embeddings for inputs.
  emb_net = tl.layers.Word2vecEmbeddingInputlayer(
          inputs = train_inputs,
          train_labels = train_labels,
          vocabulary_size = vocabulary_size,
          embedding_size = embedding_size,
          num_sampled = num_sampled,
          nce_loss_args = {},
          E_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
          E_init_args = {},
          nce_W_init = tf.truncated_normal_initializer(
                            stddev=float(1.0/np.sqrt(embedding_size))),
          nce_W_init_args = {},
          nce_b_init = tf.constant_initializer(value=0.0),
          nce_b_init_args = {},
          name ='word2vec_layer',
      )
  cost = emb_net.nce_cost

数据集迭代和损失
^^^^^^^^^^^^^^^
Word2vec使用负采样和Skip-gram模型进行训练。
噪音对比估计损失(NCE)会帮助减少损失的计算。
Skip-Gram 将文本和目标反转(Skip-Gram inverts context and targets)，尝试从目标单词预测每段文本单词。
我们使用 ``tl.nlp.generate_skip_gram_batch`` 来生成训练数据，如下：

.. code-block:: python

  cost = emb_net.nce_cost
  train_params = emb_net.all_params

  train_op = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1,
            use_locking=False).minimize(cost, var_list=train_params)

  data_index = 0
  while (step < num_steps):
    batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(
                  data=data, batch_size=batch_size, num_skips=num_skips,
                  skip_window=skip_window, data_index=data_index)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
    _, loss_val = sess.run([train_op, cost], feed_dict=feed_dict)


重载现有的嵌套矩阵
^^^^^^^^^^^^^^^^^^^^^^^^

在训练嵌套矩阵的最后，我们保存矩阵及其相应的词典。然后下一次，我们按如下可以重新载入这个矩阵和字典：

.. code-block:: python

  vocabulary_size = 50000
  embedding_size = 128
  model_file_name = "model_word2vec_50k_128"
  batch_size = None

  print("Load existing embedding matrix and dictionaries")
  all_var = tl.files.load_npy_to_any(name=model_file_name+'.npy')
  data = all_var['data']; count = all_var['count']
  dictionary = all_var['dictionary']
  reverse_dictionary = all_var['reverse_dictionary']

  tl.nlp.save_vocab(count, name='vocab_'+model_file_name+'.txt')

  del all_var, data, count

  load_params = tl.files.load_npz(name=model_file_name+'.npz')

  x = tf.placeholder(tf.int32, shape=[batch_size])
  y_ = tf.placeholder(tf.int32, shape=[batch_size, 1])

  emb_net = tl.layers.EmbeddingInputlayer(
                  inputs = x,
                  vocabulary_size = vocabulary_size,
                  embedding_size = embedding_size,
                  name ='embedding_layer')

  sess.run(tf.initialize_all_variables())

  tl.files.assign_params(sess, [load_params[0]], emb_net)


运行PTB示例
==================

Penn树图资料库(Penn TreeBank)的数据集在许多语言建模(LANGUAGE MODELING)论文熵使用，包括"Empirical Evaluation and Combination of Advanced Language Modeling Techniques"。
“Recurrent Neural Network Regularization”。它包括了929k个训练词，73K个验证词和82l个测试词。
在它的词库中，它有10K个词。

PTB示例试图展示在一个有挑战性的语言建模的任务如何训练一个递归神经网络。

给一句"I am from Imperial College London", the model can learn to predict "Imperial College London" from "from Imperial College".
换句话说，它在一篇给出前面单词的历史情况的文本中预测下一个单词。
在这种情况下， ``num_step（序列长度）`` 是3。

.. code-block:: bash

  python tutorial_ptb_lstm.py

该脚本提供三种设置(小，中，大),更大的模型有更好的性能，您可以在下面选项中选择不同的设置：

.. code-block:: python

  flags.DEFINE_string(
      "model", "small",
      "A type of model. Possible options are: small, medium, large.")


如果您选择小设置，您会看到：

.. code-block:: text

  Epoch: 1 Learning rate: 1.000
  0.004 perplexity: 5220.213 speed: 7635 wps
  0.104 perplexity: 828.871 speed: 8469 wps
  0.204 perplexity: 614.071 speed: 8839 wps
  0.304 perplexity: 495.485 speed: 8889 wps
  0.404 perplexity: 427.381 speed: 8940 wps
  0.504 perplexity: 383.063 speed: 8920 wps
  0.604 perplexity: 345.135 speed: 8920 wps
  0.703 perplexity: 319.263 speed: 8949 wps
  0.803 perplexity: 298.774 speed: 8975 wps
  0.903 perplexity: 279.817 speed: 8986 wps
  Epoch: 1 Train Perplexity: 265.558
  Epoch: 1 Valid Perplexity: 178.436
  ...
  Epoch: 13 Learning rate: 0.004
  0.004 perplexity: 56.122 speed: 8594 wps
  0.104 perplexity: 40.793 speed: 9186 wps
  0.204 perplexity: 44.527 speed: 9117 wps
  0.304 perplexity: 42.668 speed: 9214 wps
  0.404 perplexity: 41.943 speed: 9269 wps
  0.504 perplexity: 41.286 speed: 9271 wps
  0.604 perplexity: 39.989 speed: 9244 wps
  0.703 perplexity: 39.403 speed: 9236 wps
  0.803 perplexity: 38.742 speed: 9229 wps
  0.903 perplexity: 37.430 speed: 9240 wps
  Epoch: 13 Train Perplexity: 36.643
  Epoch: 13 Valid Perplexity: 121.475
  Test Perplexity: 116.716

PTB示例证明了RNN能够对语言进行建模，但是这个示例并没有做什么实际的事情。
但是，您应该浏览这个示例和 ``Understand LSTM`` 来理解RNN的基础。
在这之后，您将学习如何生成文本，如何实现翻译语言和如何使用RNN建立问题应答系统。

理解 LSTM
=============

递归神经网络(Recurrent Neural Network)
-------------------------------------------

董豪个人人为Andrey Karpathy的博客是 `Understand Recurrent Neural Network`_ 最好的材料。
读完这个之后，Colah的博客能帮你 `Understand LSTM Network`_ ，它能解决长期依赖(Long-Term Dependencies)的问题。
我们不介绍更多关于RNN的内容，在你继续之前，请阅读这些博客。

.. _fig_0601:

.. image:: my_figs/karpathy_rnn.jpeg

Image by Andrey Karpathy

同步序列的输入与输出(Synced sequence input and output)
--------------------------------------------------------------

在PTB示例中的模型是一个典型的同步序列的输入与输出，它被Karpathy 描述为
“(5) 同步序列输入与输出(例如视频分类，我们希望对视频的每一帧进行标记)。
注意，在不同长度的序列，每一种情况都没有预先指定的约束条件。因为递归转变是固定的
并且只要我们喜欢，可以被应用很多次。

模型的构建如下。首先通过查找嵌套矩阵，将词转换为词向量。
在本教程中，没有在嵌套矩阵熵进行预训练。
齐次，我们堆叠两个在嵌套层中使用退出率LSTM，LSTM层和正则化输出层。
该模型在训练过程中提供SOFTMAX输出的序列

第一层LSTM层为了和下一层的LSTM堆叠而输出[batch_size, num_steps, hidden_size]
第二层LSTM层为了后下一层的稠密层而输出 [batch_size*num_steps, hidden_size]，
然后计算每个实例的softmax输出，即n_examples = batch_size*num_steps。


要理解PTB教程，您也可以阅读 `TensorFlow PTB tutorial
<https://www.tensorflow.org/versions/r0.9/tutorials/recurrent/index.html#recurrent-neural-networks>`_ 。


.. code-block:: python

  network = tl.layers.EmbeddingInputlayer(
              inputs = x,
              vocabulary_size = vocab_size,
              embedding_size = hidden_size,
              E_init = tf.random_uniform_initializer(-init_scale, init_scale),
              name ='embedding_layer')
  if is_training:
      network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop1')
  network = tl.layers.RNNLayer(network,
              cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
              cell_init_args={'forget_bias': 0.0},
              n_hidden=hidden_size,
              initializer=tf.random_uniform_initializer(-init_scale, init_scale),
              n_steps=num_steps,
              return_last=False,
              name='basic_lstm_layer1')
  lstm1 = network
  if is_training:
      network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop2')
  network = tl.layers.RNNLayer(network,
              cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
              cell_init_args={'forget_bias': 0.0},
              n_hidden=hidden_size,
              initializer=tf.random_uniform_initializer(-init_scale, init_scale),
              n_steps=num_steps,
              return_last=False,
              return_seq_2d=True,
              name='basic_lstm_layer2')
  lstm2 = network
  if is_training:
      network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop3')
  network = tl.layers.DenseLayer(network,
              n_units=vocab_size,
              W_init=tf.random_uniform_initializer(-init_scale, init_scale),
              b_init=tf.random_uniform_initializer(-init_scale, init_scale),
              act = tl.activation.identity, name='output_layer')


数据集迭代
^^^^^^^^^^^^^^^^^

批规模可以被视为并发计算的个数。
如下面的示例所示，第一个批使用0到9学习序列。
第二个批使用10到19学习序列。
所以它忽略了9到10之间的信息。
如果我们只设置bath_size=1，它才应该改考虑0到20之间的所有信息。

这里的批规模(batch_size)的意思是不是和MNIST示例不一样。
在MNIST示例，批规模反映在每次迭代中我们认为实例是多少，
而在PTB的示例中，批规模是为加快运算速度的并行进程数。

如果批规模>1，那么有些信息会被忽视。
但是如果你的数据是足够长的(一个语料库通常有几十亿个字)，被忽略的信息将不影响最终的结果。

在PTB教程中，我们设置了批规模=20，所以，我们将数据拆分成20段。
在每一轮(epoch)的开始，我们用20段初始化(复位)20个RNN状态，然后分别遍历这20段。

训练数据将按如下方式产生：

.. code-block:: python

  train_data = [i for i in range(20)]
  for batch in tl.iterate.ptb_iterator(train_data, batch_size=2, num_steps=3):
      x, y = batch
      print(x, '\n',y)

.. code-block:: text

  ... [[ 0  1  2] <---x                       1st subset/ iteration
  ...  [10 11 12]]
  ... [[ 1  2  3] <---y
  ...  [11 12 13]]
  ...
  ... [[ 3  4  5]  <--- 1st batch input       2nd subset/ iteration
  ...  [13 14 15]] <--- 2nd batch input
  ... [[ 4  5  6]  <--- 1st batch target
  ...  [14 15 16]] <--- 2nd batch target
  ...
  ... [[ 6  7  8]                             3rd subset/ iteration
  ...  [16 17 18]]
  ... [[ 7  8  9]
  ...  [17 18 19]]

.. note::
    这个示例可以当作词嵌套矩阵的预训练。

损失和更新公式
^^^^^^^^^^^^^^^^^^^^^

成本函数是每个最小规模的平均成本。

.. code-block:: python

  def loss_fn(outputs, targets, batch_size, num_steps):
      # Returns the cost function of Cross-entropy of two sequences, implement
      # softmax internally.
      # outputs : 2D tensor [batch_size*num_steps, n_units of output layer]
      # targets : 2D tensor [batch_size, num_steps], need to be reshaped.
      # n_examples = batch_size * num_steps
      # so
      # cost is the averaged cost of each mini-batch (concurrent process).
      loss = tf.nn.seq2seq.sequence_loss_by_example(
          [outputs],
          [tf.reshape(targets, [-1])],
          [tf.ones([batch_size * num_steps])])
      cost = tf.reduce_sum(loss) / batch_size
      return cost

  # Cost for Training
  cost = loss_fn(network.outputs, targets, batch_size, num_steps)

对于更新，这个例子在几轮(由 ``max_epoch`` 定义)学习之后通过复接一个 ``Ir_decay`` 会降低学习率。
此外，截断的反向传播方法通过他们的范数的和的比例来逼近梯度的值(truncated backpropagation clips values of gradients by the ratio of the sum of
their norms),可以用来简化学习过程。

.. code-block:: python

  # Truncated Backpropagation for training
  with tf.variable_scope('learning_rate'):
      lr = tf.Variable(0.0, trainable=False)
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                    max_grad_norm)
  optimizer = tf.train.GradientDescentOptimizer(lr)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

然后在每轮的开始，我们分配一个新的学习速度：

.. code-block:: python

  new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
  sess.run(tf.assign(lr, learning_rate * new_lr_decay))

在每一个轮的开始，LSTM的所有状态需要被复位(初始化)，
然后在每次迭代中，新的最终状态需要被置顶为下一次迭代的初始状态：

.. code-block:: python

  state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
  state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
  for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data,
                                              batch_size, num_steps)):
      feed_dict = {input_data: x, targets: y,
                  lstm1.initial_state: state1,
                  lstm2.initial_state: state2,
                  }
      # For training, enable dropout
      feed_dict.update( network.all_drop )
      _cost, state1, state2, _ = sess.run([cost,
                                      lstm1.final_state,
                                      lstm2.final_state,
                                      train_op],
                                      feed_dict=feed_dict
                                      )
      costs += _cost; iters += num_steps

预测
^^^^^^^^^^^^^

在训练模型之后，我们不再考虑步长(序列的长度)，即 ``批规模，步数`` 为 ``1`` 。
然后，我们可以一步步预测下一个单词，而不是从一个词序列预测另一个词序列。


.. code-block:: python

  state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
  state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
  for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data,
                                              batch_size, num_steps)):
      feed_dict = {input_data: x, targets: y,
                  lstm1.initial_state: state1,
                  lstm2.initial_state: state2,
                  }
      # For training, enable dropout
      feed_dict.update( network.all_drop )
      _cost, state1, state2, _ = sess.run([cost,
                                      lstm1.final_state,
                                      lstm2.final_state,
                                      train_op],
                                      feed_dict=feed_dict
                                      )
      costs += _cost; iters += num_steps

之后？
------------

现在您明白了同步序列输入和输出(Synced sequence input and output)。
让我们思考写多对一(序列输入和一个输出),我们也能用 "I am from Imperial" 来正确预测下一个单词 "College"？
请您尽可能建立一个文本生成器，给一些种子词(seed words)来生成文本。
一些人甚至用多对一模型来自动生成论文！

Karpathy的博客：
"(3) Sequence input (e.g. sentiment analysis where a given sentence is
classified as expressing positive or negative sentiment). "


运行翻译示例
===================

.. code-block:: python

  python tutorial_translate.py

该脚本将训练一个神经网络来把英文翻译成法文。
如果一切正常，您将看到：
- 下载WMT英语-法语翻译数据，包括训练数据和测试数据。
- 从英语和法语的训练数据中创建词汇库文件。
- 创建符号化的训练数据和测试数据


.. code-block:: bash

  Prepare raw data
  Load or Download WMT English-to-French translation > wmt
  Training data : wmt/giga-fren.release2
  Testing data : wmt/newstest2013

  Create vocabularies
  Vocabulary of French : wmt/vocab40000.fr
  Vocabulary of English : wmt/vocab40000.en
  Creating vocabulary wmt/vocab40000.fr from data wmt/giga-fren.release2.fr
    processing line 100000
    processing line 200000
    processing line 300000
    processing line 400000
    processing line 500000
    processing line 600000
    processing line 700000
    processing line 800000
    processing line 900000
    processing line 1000000
    processing line 1100000
    processing line 1200000
    ...
    processing line 22500000
  Creating vocabulary wmt/vocab40000.en from data wmt/giga-fren.release2.en
    processing line 100000
    ...
    processing line 22500000

  ...

首先，我们从WMT'15网站上下载英语-法语翻译数据。训练数据和测试数据如下。
训练数据用于训练模型，测试数据用于XXXX。

.. code-block:: text

  wmt/training-giga-fren.tar  <-- Training data for English-to-French (2.6GB)
                                  giga-fren.release2.* are extracted from it.
  wmt/dev-v2.tgz              <-- Testing data for different language (21.4MB)
                                  newstest2013.* are extracted from it.

  wmt/giga-fren.release2.fr   <-- Training data of French   (4.57GB)
  wmt/giga-fren.release2.en   <-- Training data of English  (3.79GB)

  wmt/newstest2013.fr         <-- Testing data of French    (393KB)
  wmt/newstest2013.en         <-- Testing data of English   (333KB)

``giga-fren.release2.*`` 是训练数据，以下是 ``giga-fren.release2.fr`` ：

.. code-block:: text

  Il a transformé notre vie | Il a transformé la société | Son fonctionnement | La technologie, moteur du changement Accueil | Concepts | Enseignants | Recherche | Aperçu | Collaborateurs | Web HHCC | Ressources | Commentaires Musée virtuel du Canada
  Plan du site
  Rétroaction
  Crédits
  English
  Qu’est-ce que la lumière?
  La découverte du spectre de la lumière blanche Des codes dans la lumière Le spectre électromagnétique Les spectres d’émission Les spectres d’absorption Les années-lumière La pollution lumineuse
  Le ciel des premiers habitants La vision contemporaine de l'Univers L’astronomie pour tous
  Bande dessinée
  Liens
  Glossaire
  Observatoires
  ...

``giga-fren.release2.en`` 如下所示，我们可以看到单词或者句子用 ""|"" 或 "\n" 来分隔。

.. code-block:: text

  Changing Lives | Changing Society | How It Works | Technology Drives Change Home | Concepts | Teachers | Search | Overview | Credits | HHCC Web | Reference | Feedback Virtual Museum of Canada Home Page
  Site map
  Feedback
  Credits
  Français
  What is light ?
  The white light spectrum Codes in the light The electromagnetic spectrum Emission spectra Absorption spectra Light-years Light pollution
  The sky of the first inhabitants A contemporary vison of the Universe Astronomy for everyone
  Cartoon
  Links
  Glossary
  Observatories

测试数据 ``newstest2013.en`` 和 ``newstest2013.fr`` 如下所示：

.. code-block:: text

  newstest2013.en :
  A Republican strategy to counter the re-election of Obama
  Republican leaders justified their policy by the need to combat electoral fraud.
  However, the Brennan Centre considers this a myth, stating that electoral fraud is rarer in the United States than the number of people killed by lightning.

  newstest2013.fr :
  Une stratégie républicaine pour contrer la réélection d'Obama
  Les dirigeants républicains justifièrent leur politique par la nécessité de lutter contre la fraude électorale.
  Or, le Centre Brennan considère cette dernière comme un mythe, affirmant que la fraude électorale est plus rare aux États-Unis que le nombre de personnes tuées par la foudre.

下载数据之后，它开始创建词汇库文件。
从训练数据 ``giga-fren.release2.fr`` 和 ``giga-fren.release2.en``创建 ``vocab40000.fr`` 和 ``vocab40000.en`` 通常需要较长一段时间。
``40000`` 反映了词汇库的规模。

``vocab40000.fr`` (381KB) 按下列所示地按每行一个项地(one-item-per-line)方式存储。

.. code-block:: text

  _PAD
  _GO
  _EOS
  _UNK
  de
  ,
  .
  '
  la
  et
  des
  les
  à
  le
  du
  l
  en
  )
  d
  0
  (
  00
  pour
  dans
  un
  que
  une
  sur
  au
  0000
  a
  par

``vocab40000.en`` (344KB) 按下列所示地按每行一个项地(one-item-per-line)方式存储。

.. code-block:: text

  _PAD
  _GO
  _EOS
  _UNK
  the
  .
  ,
  of
  and
  to
  in
  a
  )
  (
  0
  for
  00
  that
  is
  on
  The
  0000
  be
  by
  with
  or
  :
  as
  "
  000
  are
  ;

然后我们开始创建英语和法语的符号化的训练数据和测试数据。这也要较长一段时间。

.. code-block:: text

  Tokenize data
  Tokenizing data in wmt/giga-fren.release2.fr  <-- Training data of French
    tokenizing line 100000
    tokenizing line 200000
    tokenizing line 300000
    tokenizing line 400000
    ...
    tokenizing line 22500000
  Tokenizing data in wmt/giga-fren.release2.en  <-- Training data of English
    tokenizing line 100000
    tokenizing line 200000
    tokenizing line 300000
    tokenizing line 400000
    ...
    tokenizing line 22500000
  Tokenizing data in wmt/newstest2013.fr        <-- Testing data of French
  Tokenizing data in wmt/newstest2013.en        <-- Testing data of English

最后，我们所有的文件如下所示：

.. code-block:: text

  wmt/training-giga-fren.tar  <-- Compressed Training data for English-to-French (2.6GB)
                                  giga-fren.release2.* are extracted from it.
  wmt/dev-v2.tgz              <-- Compressed Testing data for different language (21.4MB)
                                  newstest2013.* are extracted from it.

  wmt/giga-fren.release2.fr   <-- Training data of French   (4.57GB)
  wmt/giga-fren.release2.en   <-- Training data of English  (3.79GB)

  wmt/newstest2013.fr         <-- Testing data of French    (393KB)
  wmt/newstest2013.en         <-- Testing data of English   (333KB)

  wmt/vocab40000.fr           <-- Vocabulary of French      (381KB)
  wmt/vocab40000.en           <-- Vocabulary of English     (344KB)

  wmt/giga-fren.release2.ids40000.fr   <-- Tokenized Training data of French (2.81GB)
  wmt/giga-fren.release2.ids40000.en   <-- Tokenized Training data of English (2.38GB)

  wmt/newstest2013.ids40000.fr         <-- Tokenized Testing data of French (268KB)
  wmt/newstest2013.ids40000.en         <-- Tokenized Testing data of English (232KB)

现在，从桶(buckets)读入所有符号化的数据并且计算他们的大小。

.. code-block:: text

  Read development (test) data into buckets
  dev data: (5, 10) [[13388, 4, 949], [23113, 8, 910, 2]]
  en word_ids: [13388, 4, 949]
  en context: [b'Preventing', b'the', b'disease']
  fr word_ids: [23113, 8, 910, 2]
  fr context: [b'Pr\xc3\xa9venir', b'la', b'maladie', b'_EOS']

  Read training data into buckets (limit: 0)
    reading data line 100000
    reading data line 200000
    reading data line 300000
    reading data line 400000
    reading data line 500000
    reading data line 600000
    reading data line 700000
    reading data line 800000
    ...
    reading data line 22400000
    reading data line 22500000
  train_bucket_sizes: [239121, 1344322, 5239557, 10445326]
  train_total_size: 17268326.0
  train_buckets_scale: [0.013847375825543252, 0.09169638099257565, 0.3951164693091849, 1.0]
  train data: (5, 10) [[1368, 3344], [1089, 14, 261, 2]]
  en word_ids: [1368, 3344]
  en context: [b'Site', b'map']
  fr word_ids: [1089, 14, 261, 2]
  fr context: [b'Plan', b'du', b'site', b'_EOS']

  the num of training data in each buckets: [239121, 1344322, 5239557, 10445326]
  the num of training data: 17268326
  train_buckets_scale: [0.013847375825543252, 0.09169638099257565, 0.3951164693091849, 1.0]

开始训练符号化的桶数据之后，训练可以通过终止程序来终止。
当 ``steps_per_checkpoint = 10`` 时，您将看到：

``steps_per_checkpoint = 10``

.. code-block:: text

  Create Embedding Attention Seq2seq Model

  global step 10 learning rate 0.5000 step-time 22.26 perplexity 12761.50
    eval: bucket 0 perplexity 5887.75
    eval: bucket 1 perplexity 3891.96
    eval: bucket 2 perplexity 3748.77
    eval: bucket 3 perplexity 4940.10
  global step 20 learning rate 0.5000 step-time 20.38 perplexity 28761.36
    eval: bucket 0 perplexity 10137.01
    eval: bucket 1 perplexity 12809.90
    eval: bucket 2 perplexity 15758.65
    eval: bucket 3 perplexity 26760.93
  global step 30 learning rate 0.5000 step-time 20.64 perplexity 6372.95
    eval: bucket 0 perplexity 1789.80
    eval: bucket 1 perplexity 1690.00
    eval: bucket 2 perplexity 2190.18
    eval: bucket 3 perplexity 3808.12
  global step 40 learning rate 0.5000 step-time 16.10 perplexity 3418.93
    eval: bucket 0 perplexity 4778.76
    eval: bucket 1 perplexity 3698.90
    eval: bucket 2 perplexity 3902.37
    eval: bucket 3 perplexity 22612.44
  global step 50 learning rate 0.5000 step-time 14.84 perplexity 1811.02
    eval: bucket 0 perplexity 644.72
    eval: bucket 1 perplexity 759.16
    eval: bucket 2 perplexity 984.18
    eval: bucket 3 perplexity 1585.68
  global step 60 learning rate 0.5000 step-time 19.76 perplexity 1580.55
    eval: bucket 0 perplexity 1724.84
    eval: bucket 1 perplexity 2292.24
    eval: bucket 2 perplexity 2698.52
    eval: bucket 3 perplexity 3189.30
  global step 70 learning rate 0.5000 step-time 17.16 perplexity 1250.57
    eval: bucket 0 perplexity 298.55
    eval: bucket 1 perplexity 502.04
    eval: bucket 2 perplexity 645.44
    eval: bucket 3 perplexity 604.29
  global step 80 learning rate 0.5000 step-time 18.50 perplexity 793.90
    eval: bucket 0 perplexity 2056.23
    eval: bucket 1 perplexity 1344.26
    eval: bucket 2 perplexity 767.82
    eval: bucket 3 perplexity 649.38
  global step 90 learning rate 0.5000 step-time 12.61 perplexity 541.57
    eval: bucket 0 perplexity 180.86
    eval: bucket 1 perplexity 350.99
    eval: bucket 2 perplexity 326.85
    eval: bucket 3 perplexity 383.22
  global step 100 learning rate 0.5000 step-time 18.42 perplexity 471.12
    eval: bucket 0 perplexity 216.63
    eval: bucket 1 perplexity 348.96
    eval: bucket 2 perplexity 318.20
    eval: bucket 3 perplexity 389.92
  global step 110 learning rate 0.5000 step-time 18.39 perplexity 474.89
    eval: bucket 0 perplexity 8049.85
    eval: bucket 1 perplexity 1677.24
    eval: bucket 2 perplexity 936.98
    eval: bucket 3 perplexity 657.46
  global step 120 learning rate 0.5000 step-time 18.81 perplexity 832.11
    eval: bucket 0 perplexity 189.22
    eval: bucket 1 perplexity 360.69
    eval: bucket 2 perplexity 410.57
    eval: bucket 3 perplexity 456.40
  global step 130 learning rate 0.5000 step-time 20.34 perplexity 452.27
    eval: bucket 0 perplexity 196.93
    eval: bucket 1 perplexity 655.18
    eval: bucket 2 perplexity 860.44
    eval: bucket 3 perplexity 1062.36
  global step 140 learning rate 0.5000 step-time 21.05 perplexity 847.11
    eval: bucket 0 perplexity 391.88
    eval: bucket 1 perplexity 339.09
    eval: bucket 2 perplexity 320.08
    eval: bucket 3 perplexity 376.44
  global step 150 learning rate 0.4950 step-time 15.53 perplexity 590.03
    eval: bucket 0 perplexity 269.16
    eval: bucket 1 perplexity 286.51
    eval: bucket 2 perplexity 391.78
    eval: bucket 3 perplexity 485.23
  global step 160 learning rate 0.4950 step-time 19.36 perplexity 400.80
    eval: bucket 0 perplexity 137.00
    eval: bucket 1 perplexity 198.85
    eval: bucket 2 perplexity 276.58
    eval: bucket 3 perplexity 357.78
  global step 170 learning rate 0.4950 step-time 17.50 perplexity 541.79
    eval: bucket 0 perplexity 1051.29
    eval: bucket 1 perplexity 626.64
    eval: bucket 2 perplexity 496.32
    eval: bucket 3 perplexity 458.85
  global step 180 learning rate 0.4950 step-time 16.69 perplexity 400.65
    eval: bucket 0 perplexity 178.12
    eval: bucket 1 perplexity 299.86
    eval: bucket 2 perplexity 294.84
    eval: bucket 3 perplexity 296.46
  global step 190 learning rate 0.4950 step-time 19.93 perplexity 886.73
    eval: bucket 0 perplexity 860.60
    eval: bucket 1 perplexity 910.16
    eval: bucket 2 perplexity 909.24
    eval: bucket 3 perplexity 786.04
  global step 200 learning rate 0.4901 step-time 18.75 perplexity 449.64
    eval: bucket 0 perplexity 152.13
    eval: bucket 1 perplexity 234.41
    eval: bucket 2 perplexity 249.66
    eval: bucket 3 perplexity 285.95
  ...
  global step 980 learning rate 0.4215 step-time 18.31 perplexity 208.74
    eval: bucket 0 perplexity 78.45
    eval: bucket 1 perplexity 108.40
    eval: bucket 2 perplexity 137.83
    eval: bucket 3 perplexity 173.53
  global step 990 learning rate 0.4173 step-time 17.31 perplexity 175.05
    eval: bucket 0 perplexity 78.37
    eval: bucket 1 perplexity 119.72
    eval: bucket 2 perplexity 169.11
    eval: bucket 3 perplexity 202.89
  global step 1000 learning rate 0.4173 step-time 15.85 perplexity 174.33
    eval: bucket 0 perplexity 76.52
    eval: bucket 1 perplexity 125.97
    eval: bucket 2 perplexity 150.13
    eval: bucket 3 perplexity 181.07
  ...

经过350000轮训练模型之后，您可以将 ``main_train()`` 换为 ``main_decode()`` 来使用翻译器。
您可以输入一个英文句子，程序将输出一个法文句子。

.. code-block:: text

  Reading model parameters from wmt/translate.ckpt-340000
  >  Who is the president of the United States?
  Qui est le président des États-Unis ?


理解翻译器
====================

Seq2seq
--------------
序列到序列的模型通常被用来从一种语言到另一种语言的翻译。
但实际上它能用了做很多您可能无法想象的事情，我们可以将一个长句翻译成短且简单的句子，
例如，从莎士比亚的语言翻译成现代英语。用卷积神经网络(CNN)，我们也能将视频翻译成句子，即是视频字幕。

如果您只是想用Seq2seq，您唯一需要的时处理数据的格式，包括如何分词，如何符号化这些单词等等。
在本教程中，我们介绍了很多关于数据格式化的内容。

基础
^^^^^^^^^

序列到序列模型时一种多对多的模型，但和PTB教程中的同步序列输入与输出(Synced sequence input and output)不一样。
Seq2seq在提供所有序列输入后生成序列输出。
下列两种方法可以提高准确度：
- 反向输入
- 注意机制(Attention mechanism)

要加快计算速度，我们使用：
- softmax取样(Sampled softmax)

Karpathy的博客这样描述Seq2seq的："(4) Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French)."


.. _fig_0601:

.. image:: my_figs/basic_seq2seq.png
  :scale: 100 %
  :align: center

如上图所示，编码器输入(encoder_input)，解码器输入(decoder_input)和目标(targets)是：

.. code-block:: text

   encoder_input =  A    B    C
   decoder_input =  <go> W    X    Y    Z
   targets       =  W    X    Y    Z    <eos>

    Note：在实际的代码中，目标集的规模一个小于解码器输入的规模的数字，而不像这个数字。

论文
^^^^^^^^^^^

英语-法语的例子实现了一个作为编码器多层回归神经网络和一个基于注意(Attention-based)解码器。
它和这篇论文中描述的模型一样：
 - `Grammar as a Foreign Language <http://arxiv.org/abs/1412.7449>`_

示例采用了softmax抽样(sampled softmax)来解决大规模词汇库规模输出的问题。
在这种情况下，当 ``target_vocab_size=4000`` 并且词汇库规模小于 ``512`` 时，仅仅使用标准softmax损失可能时一种更好的主意。
softmax抽样在这篇论文的小节3中有描述：
 - `On Using Very Large Target Vocabulary for Neural Machine Translation <http://arxiv.org/abs/1412.2007>`_

依照在这篇文章的描述，逆序输入(Reversing the inputs)和多层神经元已经在序列到序列翻译模型已经被成功使用：
 - `Sequence to Sequence Learning with Neural Networks <http://arxiv.org/abs/1409.3215>`_

这篇文章描述了注意机制允许输入解码器更直接地访问输入数据：
 - `Neural Machine Translation by Jointly Learning to Align and Translate <http://arxiv.org/abs/1409.0473>`_

这篇文章提出该模型也可以用单层版本替代多层神经元来实现，但是必须要使用双向编码器(Bi-directional encoder)：
 - `Neural Machine Translation by Jointly Learning to Align and Translate <http://arxiv.org/abs/1409.0473>`_

实现
-------------

Bucketing and Padding
^^^^^^^^^^^^^^^^^^^^^^^^^

Bucketing是一种能有效处理不同长度句子的方法。
当要将英文翻译成法文的时候，在输入栏中我们将得到不同的长度 ``L1`` 的英文句子。
并且在输出栏法文句子的不同长度 ``L2`` 。
我们原则上应该为一个英文和法文句子的长度的每一对 ``(L1,L2+1)`` (由一个GO符号作为前缀) 建立seq2seq模型。

为了找到对于每一对数最接近bucket，那么如果bucket比句子大，我们只能在句子的末尾用一个特殊的PAD记号，来标记每一个句子。

我们使用几个buckets并且有效地把句子标记到最近的buckets。在这个示例中，我们使用4个buckets

.. code-block:: python

  buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

如果输入的是一个标记为 ``3`` 的英文句子,并且相应的输出是一个标记为 ``6`` 的法文句子，
那么他们将被放在第一个bucket并且把编码器和解码器的输入栏(英文句子)，输出栏分别标为 ``5``，``10`` 。
如果我们有一个标记为8的英文句子并且相应的法文句子被标记为18，那么他们将被放入 ``(20,25)`` bucket。

换句话说，bucket ``(I,O)`` 是 ``(编码器输入规模(encoder_input_size)，解码器输入规模(decoder_inputs_size))

给出一对符号化格式的 ``[["I", "go", "."], ["Je", "vais", "."]]`` ，我们把它转换为 ``(5,10)`` 。
编码器输入的训练数据  ``[PAD PAD "." "go" "I"]`` 并且解码器输入 ``[GO "Je" "vais" "." EOS PAD PAD PAD PAD PAD]`` 。
这些目标是解码器输入的一方面的转变。这些 ``目标权值(target_weights)`` 是 ``targets`` 的关键。

. code-block:: text

  bucket = (I, O) = (5, 10)
  encoder_inputs = [PAD PAD "." "go" "I"]                       <-- 5  x batch_size
  decoder_inputs = [GO "Je" "vais" "." EOS PAD PAD PAD PAD PAD] <-- 10 x batch_size
  target_weights = [1   1     1     1   0 0 0 0 0 0 0]          <-- 10 x batch_size
  targets        = ["Je" "vais" "." EOS PAD PAD PAD PAD PAD]    <-- 9  x batch_size

在此脚本中，一个句子是由一列表示，因此我们假设 ``批规模=3`` ， ``bucket=(5,10)`` ，训练数据看起来像这个样子：

.. code-block:: text

  encoder_inputs    decoder_inputs    target_weights    targets
  0    0    0       1    1    1       1    1    1       87   71   16748
  0    0    0       87   71   16748   1    1    1       2    3    14195
  0    0    0       2    3    14195   0    1    1       0    2    2
  0    0    3233    0    2    2       0    0    0       0    0    0
  3    698  4061    0    0    0       0    0    0       0    0    0
                    0    0    0       0    0    0       0    0    0
                    0    0    0       0    0    0       0    0    0
                    0    0    0       0    0    0       0    0    0
                    0    0    0       0    0    0       0    0    0
                    0    0    0       0    0    0

  where 0 : _PAD    1 : _GO     2 : _EOS      3 : _UNK

在训练过程中，解码器输入是目标，而在预测过程中，下一个解码器的输入是最后一个解码器的输出。

特别的语言标记(vocabulary symbols)，符号和数字。

在这个示例中特别的语言标记是：

.. code-block:: python

  _PAD = b"_PAD"
  _GO = b"_GO"
  _EOS = b"_EOS"
  _UNK = b"_UNK"
  PAD_ID = 0      <-- index (row number) in vocabulary
  GO_ID = 1
  EOS_ID = 2
  UNK_ID = 3
  _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

.. code-block:: text

          ID    MEANINGS
  _PAD    0     Padding, empty word
  _GO     1     1st element of decoder_inputs
  _EOS    2     End of Sentence of targets
  _UNK    3     Unknown word, words do not exist in vocabulary will be marked as 3

对于数字，创建词汇库和符号化数据集的 ``normalize_digits`` 必须是一致的，
如果是``True`` ，所有的数字将被 ``0`` 替代。比如 ``123`` 被 ``000`` 替代，``9`` 被 ``0``替代
，``1990-05`` 被 ``0000-00` 替代，然后 ``000`` ， ``0`` ， ``0000-00`` 等将在词汇库中(看 ``vocab40000.en`` )

相反的，如果是 ``False`` 的话，不同的数字将在词汇集中被找到。
那么词汇库规模将十分巨大。找到数字的正则表达式是 ``_DIGIT_RE = re.compile(br"\d")`` 。(详见 ``tl.nlp.create_vocabulary()`` 和 ``tl.nlp.data_to_token_ids()` )

对词进行拆分，正则表达式 ``_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")`` ，
这意味着使用 ``[ . , ! ? " ' : ; ) ( ]`` 并且分隔这句话，``tl.nlp.basic_tokenizer()`` 是 ``tl.nlp.create_vocabulary()`` 和  ``tl.nlp.data_to_token_ids()``。

所有的标点符号，类似于 ``. , ) (`` 在英文和法文数据库中的被全部保留下来。

softmax抽样(Sampled softmax)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

softmax抽样是处理大量词汇库输出的时降低计算开销的一种方法。
与计算大量输出的交叉熵不同的是，我们从 ``num_samples`` 的抽样中计算损失。

损失和更新函数
^^^^^^^^^^^^^^^^^
``EmbeddingAttentionSeq2seqWrapper`` 已经在SGD优化器上建立。

下一步？
------------------

您可以尝试其他应用。







成本函数
=================

TensorLayer提供一个简单的方法来创建您自己的成本函数。
下面以多层神经网络(MLP)为例：

.. code-block:: python

  network = tl.InputLayer(x, name='input_layer')
  network = tl.DropoutLayer(network, keep=0.8, name='drop1')
  network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
  network = tl.DropoutLayer(network, keep=0.5, name='drop2')
  network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
  network = tl.DropoutLayer(network, keep=0.5, name='drop3')
  network = tl.DenseLayer(network, n_units=10, act = tl.activation.identity, name='output_layer')


权重的正则化
----------------------

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

激活输出(Activation outputs)的正则化
---------------------------------------

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




易于修改
==================

修改预训练行为
-----------------------

贪婪的分层预训练方法(Greedy layer-wise pretrain)对深度神经网络的初始化来说是一个重要的任务，
同时根据不同的结构和应用，存在多种预训练的指标。

举个例子，比如 `"普通"稀疏自编码器(Vanilla Sparse Autoencoder ) <http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity>`_
能够按下列代码使用KL散度来实现。
但是对于`深度整流神经网络(Deep Rectifier Network) <http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf>`_ ,
稀疏可以通过使用激活输出的L1正则化来实现。

.. code-block:: python

  # Vanilla Sparse Autoencoder
  beta = 4
  rho = 0.15
  p_hat = tf.reduce_mean(activation_out, reduction_indices = 0)
  KLD = beta * tf.reduce_sum( rho * tf.log(tf.div(rho, p_hat))
          + (1- rho) * tf.log((1- rho)/ (tf.sub(float(1), p_hat))) )

出于这个原因，TensorLayer提供了一种简单的方法来修改或者涉及自己的预训练度量。
对于自编码器，TensorLayer使用 ``ReconLayer.__init__()`` 来定义重建层(reconstruction layer)和成本函数。
要定义自己的成本函数，只需要简单地在 ``ReconLayer.__init__()`` 中修改 ``self.cost`` 就可以了。
要创建您自己的成本表达式(cost expression)，请阅读  `Tensorflow Math <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html>`_ 。
默认情况下， ``重建层(ReconLayer)`` 只使用 ``self.train_params = self.all _params[-4:]`` 来更新前一层的偏差和权重，其中4个参数为``[W_encoder，b_encoder，W_decoder，b_decoder]``。
如果您想要更新前2层的参数，只需要修改 ``[-4:]`` 为 ``[-6:]``。

.. code-block:: python

  ReconLayer.__init__(...):
      ...
      self.train_params = self.all_params[-4:]
      ...
  	self.cost = mse + L1_a + L2_w

添加自定义层
--------------------

作为一个开发者提供有用的 ``层``。
TensorLayer的源代码很容易理解，打开 :mod:`tensorlayer/layer.py` 并且阅读 ``DenseLayer`` ，您可以完全理解它是怎么工作的。

添加自定义正则化函数
------------------------

详见 :mod:`tensorlayer/cost.py` 。










更多信息
==============

有关您用TensorLayer能做什么的信息，只要继续阅读readthedocs就能知道。

最后，参考列表和说明如下：


layers (:mod:`tensorlayer.layers`),

activation (:mod:`tensorlayer.activation`),

natural language processing (:mod:`tensorlayer.nlp`),

reinforcement learning (:mod:`tensorlayer.rein`),

cost expressions and regularizers (:mod:`tensorlayer.cost`),

load and save files (:mod:`tensorlayer.files`),

operating system (:mod:`tensorlayer.ops`),

helper functions (:mod:`tensorlayer.utils`),

visualization (:mod:`tensorlayer.visualize`),

iteration functions (:mod:`tensorlayer.iterate`),

preprocessing functions (:mod:`tensorlayer.preprocess`),


.. _Deeplearning Tutorial: http://deeplearning.stanford.edu/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _TensorFlow tutorial: https://www.tensorflow.org/versions/r0.9/tutorials/index.html
.. _Understand Deep Reinforcement Learning: http://karpathy.github.io/2016/05/31/rl/
.. _Understand Recurrent Neural Network: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. _Understand LSTM Network: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. _Word Representations: http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
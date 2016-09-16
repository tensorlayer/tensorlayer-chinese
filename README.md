<!--<div align="center">
	<div class="TensorFlow">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" style=": left; margin-left: 5px; margin-bottom: 5px;"><br><br>
   </div>
   <div class="TensorLayer">
    <img src="https://www.tensorflow.org/images/tf_logo_transp.png" style=": right; margin-left: 5px; margin-bottom: 5px;">
    </div>
</div>
-->
<a href="http://github.com/zsdonghao/tensorlayer">
<div align="center">
	<img src="img/img_tensorlayer.png" width="30%" height="30%"/>
</div>
</a>


# TensorLayer: 面向研究人员和工程师的深度学习和强化学习库  

TensorLayer 是为研究人员和工程师设计的一款基于[Google TensorFlow](https://www.tensorflow.org)开发的深度学习与强化学习库。 
它提供高级别的（Higher-Level）深度学习API，这样不仅可以加快研究人员的实验速度，也能够减少工程师在实际开发当中的重复工作。
TensorLayer非常易于修改和扩展，这使它可以同时用于机器学习的研究与应用。此外，TensorLayer 提供了大量示例和教程来帮助初学者理解深度学习，并提供大量的官方例子程序方便开发者快速找到适合自己项目的例子。

阅读TensorLayer Readthedocs 文档您不仅可以学会如何使用这个库，也会了解不同类型的神经网络、深度学习、强化学习，还有自然语言处理等内容。此外，TensorLayer的Tutorial包含了所有TensorFlow官方深度学习教程的模块化实现，因此你可以对照TensorFlow深度学习教程来学习[[英文]](https://www.tensorflow.org/versions/master/tutorials/index.html)[[极客学院中文翻译]](http://wiki.jikexueyuan.com/project/tensorflow-zh/)。

不过，与其它基于TensorFlow开发的傻瓜式API不同，TensorLayer需要使用者有基本的神经网络知识。了解TensorFlow的基础，可以让你非常熟练地使用它。

🌞🌞🌞 我们建议你在[Github](http://github.com/zsdonghao/tensorlayer) 上star和watch[官方项目](http://github.com/zsdonghao/tensorlayer)，这样当官方有更新时，你会立即知道。本文档为[官方RTD文档](https://github.com/zsdonghao/tensorlayer)的翻译版，更新速度会比英文原版慢，若你的英文还行，我们建议你直接阅读[官方RTD文档](https://github.com/zsdonghao/tensorlayer)

❤️❤️❤️ TensorLayer首批开发者包括中国人，我们承诺将一直支持中国社区

-


TensorLayer 在兼顾 TensorFlow 的灵活性的同时，又能为使用者提供合适的操作粒度来建立和训练神经网络。TensorLayer的开发遵循以下几个原则：

- 透明性：用户可以直接使用 TensorFlow 的方法来操作所有有的训练，迭代，初始化过程，我们鼓励用户尽可能多的在TensorLayer中使用TensorFlow的方法，利用TensorFlow所提供的便利。
- Tensor：张量是一个可用来表示在一些向量、标量和其他张量之间的线性关系的多线性函数。TensorFlow 使用这种数据结构来表示神经网络所需要的数据。
- 教程：TensorLayer提供了大量的连贯教程，让用户可以循序渐进的学习使用TensorLayer和深度学习了解，教程的内容覆盖了 Dropout, DropConnect, Denoising Autoencoder, LSTM, CNN 等等。
- TPU：Tensor Process Unit 是为了针对 TensorFlow 深度学习打造的定制化ASIC芯片。
- 分布式：TensorFlow 默认支持分布式系统。
- 兼容性：单层网络的建立被抽象成正则化，成本和每一层的输出，方便与其他基于TensorFlow的库协作。
- 简洁：易于使用，扩展与修改，以便在研究和工程中使用。
- 高速：在GPU的支持下运行速度与纯TensorFlow脚本速度一致。简洁但不牺牲性能。

让我们在 [overview](#overview) 中看看TensorLayer强大的功能吧!!!

# 安装

TensorLayer has install prerequisites including TensorFlow, numpy and matplotlib. For GPU support, CUDA and cuDNN are required. Please check [here](http://tensorlayer.readthedocs.io/en/latest/user/installation.html) for detailed instructions.

If you already had the pre-requisites ready, the simplest way to install TensorLayer in your python program is: 

```python
pip install tensorlayer
or
pip install git+https://github.com/zsdonghao/tensorlayer.git
```

# 您第一个程序

第一个程序训练一个多层神经网络来对 MNIST 阿拉伯数字图片集进行分类。我们使用了简单了 [scikit](http://scikit-learn.org/stable/)式函数，如 ``fit()`` 和 ``test()`` 。 

```python
import tensorflow as tf
import tensorlayer as tl
import time

sess = tf.InteractiveSession()

# 准备数据
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))

# 定义 placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# 定义模型
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')

# 定义输出、损失函数和衡量指标
# tl.cost.cross_entropy 在内部使用 tf.nn.sparse_softmax_cross_entropy_with_logits() 实现 softmax
network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity, name='output_layer')
y = network.outputs
cost = tl.cost.cross_entropy(y, y_)
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# 定义 optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# 初始化所有参数
sess.run(tf.initialize_all_variables())

# 列出模型信息
network.print_params()
network.print_layers()

# 训练模型, 但如果你想了解更多实现细节，或想成为机器学习领域的专家，
# 我们鼓励使用 tl.iterate.minibatches() 来训练模型
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=500, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# 评估模型
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# 把模型保存成 .npz 文件
tl.files.save_npz(network.all_params , name='model.npz')

sess.close()
```

我们提供简单的 APIs 如 `fit()` , `test()`，这和 Scikit-learn, Keras 很相识，都是为了加快编程速度。不过，如果您想更好地控制训练过程，您可以在您的代码中使用 TensorFlow 原本的方法，如 `sess.run` （`tutorial_mnist.py` 提供了一些例子）。更多关于DL和RL的例子，请见 [这里](http://tensorlayer.readthedocs.io/en/latest/user/example.html)。

# 文档

文档  [[CN]](http://tensorlayercn.readthedocs.io) [[EN]](http://tensorlayer.readthedocs.io) [[PDF]](https://media.readthedocs.org/pdf/tensorlayer/latest/tensorlayer.pdf) [[Epub]](http://readthedocs.org/projects/tensorlayer/downloads/epub/latest/) [[HTML]](http://readthedocs.org/projects/tensorlayer/downloads/htmlzip/latest/) 不仅描述了如何使用 TensorLayer APIs，它还提供了大量的深度学习教程包括不同类型的神经网络、增强学习、自然语言处理等等。

我们还提供了 Google TensorFlow 网站例子的模块化实现，因此您还可以同时阅读 TensorFlow 的例子教程 [[en]](https://www.tensorflow.org/versions/master/tutorials/index.html) [[cn]](http://wiki.jikexueyuan.com/project/tensorflow-zh/) 。



# 贡献指南

TensorLayer 起初是帝国理工大学的内部项目，用来帮助研究人员验证新的算法。现在它鼓励全世界的人工智能爱好者们参与开发，以促进学术和应用交流。您可以和我们联系讨论您的想法，或者在官方 Github 上发起 Fork 与 Pull 请求。

- 🇬🇧 If you are in London, we can discuss in person
- 🇨🇳 我们有官方的 [中文文档](http://tensorlayercn.readthedocs.io/zh/latest/). 与此同时, 我们建立了多种交流渠道，如[QQ 群](img/img_qq.png) ，加入微信群需要您可把个人介绍和微信号发送到 haodong_cs@163.com 申请
- 🇹🇭 เราขอเรียนเชิญนักพัฒนาคนไทยทุกคนที่สนใจจะเข้าร่วมทีมพัฒนา TensorLayer ติดต่อสอบถามรายละเอียดเพิ่มเติมได้ที่ haodong_cs@163.com

# 版权

TensorLayer is releazed under the Apache 2.0 license.


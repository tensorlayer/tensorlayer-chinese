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

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Help Wanted Issues](https://badge.waffle.io/zsdonghao/tensorlayer.svg?label=up-for-grabs&title=Help Wanted Issues)](https://waffle.io/zsdonghao/tensorlayer)

TensorLayer 是基于 [Google TensorFlow](https://www.tensorflow.org) 开发的深度学习与增强学习库。它提供主流的深度学习与增强学习模块，可以非常容易地自定义模型以解决人工智能问题。

TensorLayer grow out from a need to combine the power of TensorFlow with the right building modules for deep neural networks. According to our years of research and practical experiences of tackling real-world machine learning problems, we come up with three design goals for TensorLayer:

- **Simplicity**: we make TensorLayer easy to work with by providing mass tutorials that can be deployed and run through in minutes. A TensorFlow user may find it easier to bootstrap with the simple, high-level APIs provided by TensorLayer, and then deep dive into their implementation details if need. 
- **Flexibility**: developing an effective DL algorithm for a specific domain typically requires careful tunings from many aspects. Without the loss of simplicity, TensorLayer allows users to customize their modules by manipulating the native APIs of TensorFlow (e.g., training parameters, iteration control and tensor components).
- **Performance**: TensorLayer aims to provide zero-cost abstraction for TensorFlow. With its first-class support for TensorFlow, it can easily run on either heterogeneous platforms or multiple computation nodes without compromise in performance.

关于 TensorLayer 一个最常见的问题就是为什么我们需要开发这个库，与其他库如 [Keras](https://github.com/fchollet/keras) 和 [Tflearn](https://github.com/tflearn/tflearn)有什么区别。
TensorLayer 和这些库最大的区别在于灵活性和运行速度。深度学习用户会发现使用 Keras 和 Tflearn 能够非常快的上手（当然 TensorLayer 也提供与它们类似的简单 APIs），这些库提供高层抽象的API，对开发者隐藏了深度学习引擎的细节。这会让用户很难从底层中修改和优化，而这往往在特定领域时需要考虑的。尽管如此，灵活性不会导致效率的降低，TensorLayer 可以分布式和多样化部署以最优化运行速度。此外，TensorLayer 还能和很多库无缝使用，如 [TF-Slim](https://github.com/tensorflow/models/tree/master/slim) 等等。

# 译者注

简单来讲 TensorLayer 是一个适用于不同水平用户使用的库。对于初学者，TensorLayer 提供大量简单的API和大量的教程；对于中级用户，TensorLayer 的灵活性和透明性优势能大大体现出来（V1.2版本是很好的例子）；对于高级用户，运行速度快和跨平台优势会体现出来。这样的好处是作为用户，我们不需要因为在不同的学习阶段，而去学不同的库了。

🌞🌞🌞 我们建议你在[Github](http://github.com/zsdonghao/tensorlayer) 上star和watch[官方项目](http://github.com/zsdonghao/tensorlayer)，这样当官方有更新时，你会立即知道。本文档为[官方RTD文档](https://github.com/zsdonghao/tensorlayer)的翻译版，更新速度会比英文原版慢，若你的英文还行，我们建议你直接阅读[官方RTD文档](https://github.com/zsdonghao/tensorlayer)

❤️❤️❤️ TensorLayer首批开发者包括中国人，我们承诺将一直支持中国社区


# 安装

TensorLayer 运行需要 TensorFlow, numpy 和 matplotlib。 对于 GPU 加速，需要安装 CUDA 和 cuDNN。请在 [这里](http://tensorlayercn.readthedocs.io/zh/latest/user/installation.html) 查看更多安装细节。

如果您已经安装过 TensorFlow，最简单的安装命令如下 (以英文[Github](http://github.com/zsdonghao/tensorlayer)为准)：

```bash
[for stable version] pip install tensorlayer==1.2.2b    
[for master version] pip install git+https://github.com/zsdonghao/tensorlayer.git
```

# 您第一个程序

第一个程序训练一个多层神经网络来对 MNIST 阿拉伯数字图片集进行分类。我们使用了简单了 [scikit](http://scikit-learn.org/stable/)式函数，如 ``fit()`` 和 ``test()`` 。 

```python
import tensorflow as tf
import tensorlayer as tl

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

<!--
TensorLayer 起初是帝国理工大学的内部项目，用来帮助研究人员验证新的算法。现在它鼓励全世界的人工智能爱好者们参与开发，以促进学术和应用交流。您可以和我们联系讨论您的想法，或者在官方 Github 上发起 Fork 与 Pull 请求。
 -->
TensorLayer is a major ongoing research project in Data Science Institute, Imperial College London.
TensorLayer contributors are from Imperial College, Tsinghua University, Carnegie Mellon University, Google, Microsoft, Bloomberg and etc.
The goal of the project is to develop a compositional language while complex learning systems
can be build through composition of neural network modules.
The whole development is now participated by numerous contributors [here](https://github.com/zsdonghao/tensorlayer/releases).


- 🇬🇧 If you are in London, we can discuss in person. Drop us an email to organize a meetup: tensorlayer@gmail.com.
- 🇨🇳 我们有官方的 [中文文档](http://tensorlayercn.readthedocs.io/zh/latest)。另外, 我们建立了多种交流渠道，如[QQ 群](img/img_qq.png)和微信群*（申请入群时请star该项目，并告知github用户名）*. 需加入微信群，请将个人介绍和微信号发送到 tensorlayer@gmail.com.
- 🇹🇭 เราขอเรียนเชิญนักพัฒนาคนไทยทุกคนที่สนใจจะเข้าร่วมทีมพัฒนา TensorLayer ติดต่อสอบถามรายละเอียดเพิ่มเติมได้ที่ tensorlayer@gmail.com.

# 版权

TensorLayer is releazed under the Apache 2.0 license.


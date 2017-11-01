TensorLayer 中文文档
==============================


.. image:: user/my_figs/img_tensorlayer.png
  :scale: 25 %
  :align: center
  :target: https://github.com/zsdonghao/tensorlayer

**好消息！我们获得了**`**ACM Multimedia (MM)** <http://www.acmmm.org/2017/mm-2017-awardees/>`_**年度最佳开源软件奖，上一次获得该奖的深度学习软件是**`**Caffe** <https://github.com/BVLC/caffe>`_**。**

TensorLayer 是为研究人员和工程师设计的一款基于Google TensorFlow开发的深度学习与强化学习库。
它提供高级别的（Higher-Level）深度学习API，这样不仅可以加快研究人员的实验速度，也能够减少工程师在实际开发当中的重复工作。 TensorLayer非常易于修改和扩展，这使它可以同时用于机器学习的研究与应用。
此外，TensorLayer 提供了大量示例和教程来帮助初学者理解深度学习，并提供大量的官方例子程序方便开发者快速找到适合自己项目的例子。

这篇文档不仅仅是为了描述如何使用这个库也是一个遍历不同类型的神经网络，
深度强化学习和自然语言处理等内容的教程。
此外，TensorLayer的Tutorial包含了所有TensorFlow官方深度学习教程的模块化实现，因此你可以对照TensorFlow深度学习教程来学习 `[英文] <https://www.tensorflow.org/versions/master/tutorials/index.html>`_ `[极客学院中文翻译] <http://wiki.jikexueyuan.com/project/tensorflow-zh/>`_

.. note::
  我们建议你在 `Github <https://github.com/zsdonghao/tensorlayer>`_ 上star和watch `官方项目 <https://github.com/zsdonghao/tensorlayer>`_ ，这样当官方有更新时，你会立即知道。本文档为 `官方RTD文档 <http://tensorlayer.readthedocs.io/>`_ 的翻译版，更新速度会比英文原版慢，若你的英文还行，我们建议你直接阅读 `官方RTD文档 <http://tensorlayer.readthedocs.io/>`_。

  如果你阅读在线中文文档时有什么问题，你可以在github上下载这个项目，
  然后去 ``/docs/cn/_build/html/index.html`` 阅读离线中文文档。
  或者在 `Read the docs <http://tensorlayer.readthedocs.io/en/latest/>`_ 中阅读官方原文档。

用户指南
-----------

TensorLayer用户指南说明了如何去安装TensorFlow,CUDA和cuDNN，
然后如何用TensorLayer建立和训练神经网络和如何作为（一个开发者支持这个库。）

.. toctree::
   :maxdepth: 2

   user/installation
   user/tutorial
   user/example
   user/development
   user/more

API目录
----------

如果你正在寻找某个特殊的函数，类或者方法，这一列文档就是为你准备的。

.. toctree::
  :maxdepth: 2

  modules/layers
  modules/cost
  modules/prepro
  modules/iterate
  modules/utils
  modules/nlp
  modules/rein
  modules/files
  modules/visualize
  modules/ops
  modules/activation
  modules/db


索引与附录
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/zsdonghao/tensorlayer

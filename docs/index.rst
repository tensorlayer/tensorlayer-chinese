欢迎来到 TensorLayer
==============================


.. image:: user/my_figs/img_tensorlayer.png
  :scale: 25 %
  :align: center
  :target: https://github.com/zsdonghao/tensorlayer


TensorLayer 是一个透明的基于Google TensorFlow顶层的透明化深度学习与强化学习库。
它被设计为了加快实验速度而提供更高水平的TensorFlow API。
TensorLayer易于扩展和修改，同时也适用于机器学习的研究与应用。

这篇文档不仅仅是为了描述如何使用这个库也是一个遍历不同类型的神经网络，
深度强化学习和自然语言处理等内容的教程。

.. note::
   本文档为 `TensorLayer Github <https://github.com/zsdonghao/tensorlayer>`_
   的翻译版，更新速度会比英文原版慢，若你的英文很好，我们建议你直接阅读 `英文文档 <http://tensorlayer.readthedocs.io/>`_。

   如果你阅读在线中文文档时有什么问题，你可以在github上下载这个项目，
   然后去``/docs/cn/_build/html/index.html``阅读离线中文文档。
   或者在 `TensorLayer <http://tensorlayer.readthedocs.io/en/latest/>`_ 中阅读官方原文档。

用户指南
-----------

TensorLayer用户指南说明了如何去安装TensorFlow,CUDA和cuDNN，
然后如何用TensorLayer建立和训练神经网络和如何作为（一个开发者支持这个库。）

.. toctree::
   :maxdepth: 2

   user/installation
   user/tutorial
   user/development

API目录
----------

如果你正在寻找某个特殊的函数，类或者方法，这一列文档就是为你准备的。

.. toctree::
  :maxdepth: 2

  modules/layers
  modules/activation
  modules/nlp
  modules/rein
  modules/iterate
  modules/cost
  modules/visualize
  modules/files
  modules/utils
  modules/init
  modules/preprocess
  modules/ops

索引与附录
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/zsdonghao/tensorlayer



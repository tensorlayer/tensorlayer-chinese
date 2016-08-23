API - 数据与模型的可视化
============================

TensorFlow 提供了可视化模型和激活输出等的工具 `TensorBoard <https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html>`_。
在这里，我们进一步提供一些可视化模型参数和数据的函数。

.. automodule:: tensorlayer.visualize

.. autosummary::

   W
   CNN2d
   frame
   images2d
   tsne_embedding

可视化权值
--------------------

.. autofunction:: W

.. autofunction:: CNN2d

.. autofunction:: frame

.. autofunction:: images2d

可视化词向量矩阵
--------------------

.. autofunction:: tsne_embedding

API - 数据与模型的可视化
============================

TensorFlow 提供了可视化模型和激活输出等的工具 `TensorBoard <https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html>`_。
在这里，我们进一步提供一些可视化模型参数和数据的函数。

.. automodule:: tensorlayer.visualize

.. autosummary::

   read_image
   read_images
   save_image
   save_images
   W
   CNN2d
   frame
   images2d
   tsne_embedding

读取与保存图片
--------------

读取单个图片
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: read_image

读取多个图片
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: read_images

保存单个图片
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_image

保存多个图片
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_images

可视化模型参数
--------------------

可视化Weight Matrix
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: W

可视化CNN 2d filter
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: CNN2d


可视化图像
----------------

matplotlib显示单图
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: frame

matplotlib显示多图
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: images2d

可视化词嵌入矩阵
--------------------
.. autofunction:: tsne_embedding

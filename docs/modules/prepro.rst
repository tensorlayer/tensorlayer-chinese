API - 数据预处理
=============================

数据预处理，更多关于图像、信号的Tensor函数，可在 `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_ 中找到。

.. automodule:: tensorlayer.prepro

.. autosummary::

   distorted_images
   crop_central_whiten_images


图像预处理
--------------------

对训练数据
^^^^^^^^^^^^^^^^
.. autofunction:: distorted_images

对测试数据
^^^^^^^^^^^^^^^^^
.. autofunction:: crop_central_whiten_images

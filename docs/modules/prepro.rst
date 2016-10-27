API - 数据预处理
=============================

我们提供大量的数据增强及处理方法，使用 Numpy, Scipy, Threading 和 Queue。
不过，我们建议你直接使用 TensorFlow 提供的 operator，如 ``tf.image.central_crop`` ，更多关于 TensorFlow 的信息请见
`这里 <https://www.tensorflow.org/versions/master/api_docs/python/image.html>`_ 和 ``tutorial_cifar10_tfrecord.py``.
这个包的一部分代码来自Keras。


.. automodule:: tensorlayer.prepro

.. autosummary::

   threading_data

   rotation
   rotation_multi
   crop
   crop_multi
   flip_axis
   flip_axis_multi
   shift
   shift_multi
   shear
   shear_multi
   zoom
   zoom_multi
   channel_shift
   channel_shift_multi
   transform_matrix_offset_center
   apply_transform
   array_to_img

   pad_sequences

   distorted_images
   crop_central_whiten_images


并行 Threading
------------------

.. autofunction:: threading_data


图片
-----------

- All functions have argument ``is_random``.
- All functions end with `multi` , usually be used for image segmentation i.e. the input and output image should be matched.

旋转
^^^^^^^^^
.. autofunction:: rotation
.. autofunction:: rotation_multi

裁剪
^^^^^^^^^
.. autofunction:: crop
.. autofunction:: crop_multi

颠倒
^^^^^^^^^
.. autofunction:: flip_axis
.. autofunction:: flip_axis_multi

位移
^^^^^^^^^
.. autofunction:: shift
.. autofunction:: shift_multi

切变
^^^^^^^^^
.. autofunction:: shear
.. autofunction:: shear_multi

缩放
^^^^^^^^^
.. autofunction:: zoom
.. autofunction:: zoom_multi

通道位移
^^^^^^^^^^^^^^
.. autofunction:: channel_shift
.. autofunction:: channel_shift_multi

手动变换
^^^^^^^^^^^^^^^^^
.. autofunction:: transform_matrix_offset_center
.. autofunction:: apply_transform


Numpy 与 PIL
^^^^^^^^^^^^^^
.. autofunction:: array_to_img




序列
---------

更多相关函数，请见 ``tensorlayer.nlp`` 。

.. autofunction:: pad_sequences





Tensor Opt
------------

这几个函数将被弃用， 关于如何使用 Tensor Operator 请参考 ``tutorial_cifar10_tfrecord.py`` 。

.. autofunction:: distorted_images
.. autofunction:: crop_central_whiten_images

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
   shear2
   shear_multi2
   swirl
   swirl_multi
   elastic_transform
   elastic_transform_multi

   zoom
   zoom_multi

   brightness
   brightness_multi

   illumination

   imresize

   samplewise_norm
   featurewise_norm

   channel_shift
   channel_shift_multi

   drop

   transform_matrix_offset_center
   apply_transform
   projective_transform_by_points

   array_to_img

   find_contours
   pt2map
   binary_dilation
   dilation
   binary_erosion
   erosion

   pad_sequences
   remove_pad_sequences
   process_sequences
   sequences_add_start_id
   sequences_add_end_id
   sequences_add_end_id_after_pad
   sequences_get_mask


并行 Threading
------------------

.. autofunction:: threading_data


图像
-----------

- 这些函数只对一个图像做处理， 使用 ``threading_data`` 函数来实现多线程处理，请参考 ``tutorial_image_preprocess.py`` 。
- 所有函数都有一个 ``is_random`` 。
- 所有结尾是 `multi` 的函数通常用于图像分隔，因为输入和输出的图像必需是匹配的。

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

切变 V2
^^^^^^^^^
.. autofunction:: shear2
.. autofunction:: shear_multi2

漩涡
^^^^^^^^^
.. autofunction:: swirl
.. autofunction:: swirl_multi

局部扭曲(Elastic transform)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: elastic_transform
.. autofunction:: elastic_transform_multi

缩放
^^^^^^^^^
.. autofunction:: zoom
.. autofunction:: zoom_multi

亮度
^^^^^^^^^^^^
.. autofunction:: brightness
.. autofunction:: brightness_multi

照度
^^^^^^^^^^^^^^
.. autofunction:: illumination

调整大小
^^^^^^^^^^^^
.. autofunction:: imresize

正规化
^^^^^^^^^^^^^^^
.. autofunction:: samplewise_norm
.. autofunction:: featurewise_norm

通道位移
^^^^^^^^^^^^^^
.. autofunction:: channel_shift
.. autofunction:: channel_shift_multi

噪声
^^^^^^^^^^^^^^
.. autofunction:: drop

矩阵圆心转换到图中央
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: transform_matrix_offset_center

基于矩阵的仿射变换
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: apply_transform

基于坐标点的的投影变换
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: projective_transform_by_points


Numpy 与 PIL
^^^^^^^^^^^^^^
.. autofunction:: array_to_img

找轮廓
^^^^^^^^
.. autofunction:: find_contours

一列点到图
^^^^^^^^^^^^^^^^^
.. autofunction:: pt2map

二值膨胀
^^^^^^^^^^^^^^^^^
.. autofunction:: binary_dilation

灰度膨胀
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: dilation

二值腐蚀
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: binary_erosion

灰度腐蚀
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: erosion


序列
---------

更多相关函数，请见 ``tensorlayer.nlp`` 。

Padding
^^^^^^^^^^
.. autofunction:: pad_sequences


Remove Padding
^^^^^^^^^^^^^^^^^
.. autofunction:: remove_pad_sequences

Process
^^^^^^^^^
.. autofunction:: process_sequences

Add Start ID
^^^^^^^^^^^^^^^
.. autofunction:: sequences_add_start_id


Add End ID
^^^^^^^^^^^^^^^
.. autofunction:: sequences_add_end_id

Add End ID after pad
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: sequences_add_end_id_after_pad


Get Mask
^^^^^^^^^
.. autofunction:: sequences_get_mask



Tensor Opt
------------

.. note::
  这几个函数将被弃用， 关于如何使用 Tensor Operator 请参考 ``tutorial_cifar10_tfrecord.py`` 。

.. autofunction:: distorted_images
.. autofunction:: crop_central_whiten_images

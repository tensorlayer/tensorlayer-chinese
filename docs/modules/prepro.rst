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


   obj_box_coord_rescale
   obj_box_coords_rescale
   obj_box_coord_scale_to_pixelunit
   obj_box_coord_centroid_to_upleft_butright
   obj_box_coord_centroid_to_upleft
   obj_box_coord_upleft_to_centroid

   parse_darknet_ann_str_to_list
   parse_darknet_ann_list_to_cls_box

   obj_box_left_right_flip
   obj_box_imresize
   obj_box_crop
   obj_box_shift
   obj_box_zoom



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

翻转
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



目标检测
-------------------

教程-图像增强
^^^^^^^^^^^^^^^^^^^^^^^

您好，这是基于VOC数据集的一个图像增强例子。

.. code-block:: python

  ## download the VOC dataset
  imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list, \
      classes, classes_in_person, classes_dict,\
      n_objs_list, objs_info_list, objs_info_dicts = tl.files.load_voc_dataset(dataset="2012", contain_classes_in_person=False)

  ## parse the annotation into list format
  ann_list = []
  for info in objs_info_list:
      ann = tl.prepro.parse_darknet_ann_str_to_list(info)
      c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
      ann_list.append([c, b])

  ## different types of image augmentation
  image = tl.vis.read_image(imgs_file_list[idx])
  tl.vis.draw_boxes_and_labels_to_image(image, ann_list[idx][0], ann_list[idx][1], [], classes, True, save_name='_im_original.png')

  im_flip, coords = tl.prepro.obj_box_left_right_flip(image, coords=ann_list[idx][1], is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_flip, ann_list[idx][0], coords, [], classes, True, save_name='_im_flip.png')

  im_resize, coords = tl.prepro.obj_box_imresize(image, coords=ann_list[idx][1], size=[300, 200], is_rescale=True)
  tl.vis.draw_boxes_and_labels_to_image(im_resize, ann_list[idx][0], coords, [], classes, True, save_name='_im_resize.png')

  im_crop, clas, coords = tl.prepro.obj_box_crop(image, classes=ann_list[idx][0], coords=ann_list[idx][1], wrg=200, hrg=200, is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_crop, clas, coords, [], classes, True, save_name='_im_crop.png')

  im_shfit, clas, coords = tl.prepro.obj_box_shift(image, classes=ann_list[idx][0], coords=ann_list[idx][1], wrg=0.1, hrg=0.1, is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_shfit, clas, coords, [], classes, True, save_name='_im_shift.png')

  im_zoom, clas, coords = tl.prepro.obj_box_zoom(image, classes=ann_list[idx][0], coords=ann_list[idx][1], zoom_range=(1.3, 0.7), is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_zoom, clas, coords, [], classes, True, save_name='_im_zoom.png')

坐标-像素单位到比例单位
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_rescale

坐标-像素单位到比例单位 （多个坐标）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coords_rescale

坐标-比例单位到像素单位
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_scale_to_pixelunit

坐标-[x_center, x_center, w, h]到左上-右下单位
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_centroid_to_upleft_butright

坐标-[x_center, x_center, w, h]到左上-高宽单位
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_centroid_to_upleft

坐标-左上-高宽单位到[x_center, x_center, w, h]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_upleft_to_centroid

Darknet格式-字符转列表
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: parse_darknet_ann_str_to_list

Darknet格式-分开列表的类别和坐标
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: parse_darknet_ann_list_to_cls_box

图像-翻转
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_left_right_flip

图像-调整大小
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_imresize

图像-裁剪
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_crop

图像-位移
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction::  obj_box_shift

图像-缩放
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_zoom



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

API - 实用函数
========================

.. automodule:: tensorlayer.utils

.. autosummary::

   fit
   test
   predict
   evaluation
   class_balancing_oversample
   get_random_int
   dict_to_one
   list_string_to_dict
   flatten_list
   exit_tensorflow
   open_tensorboard
   clear_all_placeholder_variables
   set_gpu_fraction

训练、测试及预测
-----------------------

训练
^^^^^^
.. autofunction:: fit

测试
^^^^^^^
.. autofunction:: test

预测
^^^^^^
.. autofunction:: predict

评估函数
---------------------
.. autofunction:: evaluation


类平衡函数(class balancing)
----------------------------
.. autofunction:: class_balancing_oversample


随机函数
----------------------------
.. autofunction:: get_random_int

字典与列表
--------------------

设字典内容全为一
^^^^^^^^^^^^^^^^
.. autofunction:: dict_to_one

一列字符转字典
^^^^^^^^^^^^^^^^
.. autofunction:: list_string_to_dict

拉平列表
^^^^^^^^^^
.. autofunction:: flatten_list

退出 TF session 和相关进程
----------------------------
.. autofunction:: exit_tensorflow

打开 TensorBoard
----------------
.. autofunction:: open_tensorboard

清空 TensorFlow placeholder
----------------------------
.. autofunction:: clear_all_placeholder_variables

设置 GPU 使用比例
-----------------
.. autofunction:: set_gpu_fraction

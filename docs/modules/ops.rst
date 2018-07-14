API - 操作系统管理
======================

系统操作，更多函数可在 `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_ 中找到。

.. automodule:: tensorlayer.ops

.. autosummary::

   exit_tf
   open_tb
   clear_all
   set_gpu_fraction
   disable_print
   enable_print
   suppress_stdout
   get_site_packages_directory
   empty_trash

TensorFlow 操作函数
---------------------------

中断 Nvidia 进程
^^^^^^^^^^^^^^^^^
.. autofunction:: exit_tf

打开 TensorBoard
^^^^^^^^^^^^^^^^^^^
.. autofunction:: open_tb

删除 placeholder
^^^^^^^^^^^^^^^^^^
.. autofunction:: clear_all

GPU 配置函数
---------------------------
.. autofunction:: set_gpu_fraction

命令窗口显示
------------------

禁止 print
^^^^^^^^^^^^
.. autofunction:: disable_print

允许 print
^^^^^^^^^^^^
.. autofunction:: enable_print

临时禁止 print
^^^^^^^^^^^^^^^^
.. autofunction:: suppress_stdout

Site packages 信息
--------------------

.. autofunction:: get_site_packages_directory

垃圾管理
--------------------
.. autofunction:: empty_trash

API - 加载保存模型与数据
========================

下载基准(benchmark)数据集，保存加载模型和数据。


.. automodule:: tensorlayer.files

.. autosummary::

   load_mnist_dataset
   load_cifar10_dataset
   load_ptb_dataset
   load_matt_mahoney_text8_dataset
   load_imbd_dataset
   load_nietzsche_dataset
   load_wmt_en_fr_dataset

   save_npz
   load_npz
   assign_params

   save_any_to_npy
   load_npy_to_any

   npz_to_W_pdf

   load_file_list

下载数据集
------------------------

.. autofunction:: load_mnist_dataset
.. autofunction:: load_cifar10_dataset
.. autofunction:: load_ptb_dataset
.. autofunction:: load_matt_mahoney_text8_dataset
.. autofunction:: load_imbd_dataset
.. autofunction:: load_nietzsche_dataset
.. autofunction:: load_wmt_en_fr_dataset


保存与加载模型
----------------------

.. autofunction:: save_npz
.. autofunction:: load_npz
.. autofunction:: assign_params

保存与加载数据
------------------------
.. autofunction:: save_any_to_npy
.. autofunction:: load_npy_to_any


可视化 npz 文件
----------------------
.. autofunction:: npz_to_W_pdf


辅助函数
------------------

.. autofunction:: load_file_list

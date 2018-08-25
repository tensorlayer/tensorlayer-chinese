API - 文件
========================

下载基准(benchmark)数据集，保存加载模型和数据。
TensorFlow提供 ``.ckpt`` 文件格式来保存和加载模型，但为了更好地实现跨平台，
我们建议使用python标准文件格式 ``.npz`` 来保存和加载模型。

.. code-block:: python

  ## 保存模型为 .ckpt
  saver = tf.train.Saver()
  save_path = saver.save(sess, "model.ckpt")
  # 从 .ckpt 加载模型
  saver = tf.train.Saver()
  saver.restore(sess, "model.ckpt")

  ## 保存模型为 .npz
  tl.files.save_npz(network.all_params , name='model.npz')
  # 从 .npz 加载模型 (方法1)
  load_params = tl.files.load_npz(name='model.npz')
  tl.files.assign_params(sess, load_params, network)
  # 从 .npz 加载模型 (方法2)
  tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

  ## 此外，你可以这样加载预训练的参数
  # 加载第一个参数
  tl.files.assign_params(sess, [load_params[0]], network)
  # 加载前三个参数
  tl.files.assign_params(sess, load_params[:3], network)


.. automodule:: tensorlayer.files

.. autosummary::

   load_mnist_dataset
   load_fashion_mnist_dataset
   load_cifar10_dataset
   load_cropped_svhn
   load_ptb_dataset
   load_matt_mahoney_text8_dataset
   load_imdb_dataset
   load_nietzsche_dataset
   load_wmt_en_fr_dataset
   load_flickr25k_dataset
   load_flickr1M_dataset
   load_cyclegan_dataset
   load_celebA_dataset
   load_voc_dataset
   load_mpii_pose_dataset
   download_file_from_google_drive

   save_npz
   load_npz
   assign_params
   load_and_assign_npz
   save_npz_dict
   load_and_assign_npz_dict
   save_graph
   load_graph
   save_graph_and_params
   load_graph_and_params
   save_ckpt
   load_ckpt

   save_any_to_npy
   load_npy_to_any

   file_exists
   folder_exists
   del_file
   del_folder
   read_file
   load_file_list
   load_folder_list
   exists_or_mkdir
   maybe_download_and_extract

   natural_keys

   npz_to_W_pdf

下载数据集
------------------------

MNIST
^^^^^^^
.. autofunction:: load_mnist_dataset

Fashion-MNIST
^^^^^^^^^^^^^^^^
.. autofunction:: load_fashion_mnist_dataset

CIFAR-10
^^^^^^^^^^^^
.. autofunction:: load_cifar10_dataset

SVHN
^^^^^^^^^^^^
.. autofunction:: load_cropped_svhn

Penn TreeBank (PTB)
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_ptb_dataset

Matt Mahoney's text8
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_matt_mahoney_text8_dataset

IMBD
^^^^^^^
.. autofunction:: load_imdb_dataset

Nietzsche
^^^^^^^^^^^^^^
.. autofunction:: load_nietzsche_dataset

WMT'15 Website 的英文译法文数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: load_wmt_en_fr_dataset

Flickr25k
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_flickr25k_dataset

Flickr1M
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_flickr1M_dataset

CycleGAN
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_cyclegan_dataset

CelebA
^^^^^^^^^
.. autofunction:: load_celebA_dataset

VOC 2007/2012
^^^^^^^^^^^^^^^^
.. autofunction:: load_voc_dataset

MPII
^^^^^^
.. autofunction:: load_mpii_pose_dataset

Google Drive
^^^^^^^^^^^^^^^^
.. autofunction:: download_file_from_google_drive

保存与加载模型
----------------------

以列表保存模型到 .npz
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_npz

从save_npz加载模型参数列表
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_npz

把模型参数载入模型
^^^^^^^^^^^^^^^^^^^
.. autofunction:: assign_params

从.npz中加载参数并导入模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_and_assign_npz


以字典保存模型到 .npz
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_npz_dict

从save_npz_dict加载模型参数列表
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_and_assign_npz_dict


保存模型结构
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_graph
        
加载模型结构
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_graph

保存模型结构和参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_graph_and_params

加载模型结构和参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_graph_and_params

以列表保存模型到 .ckpt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_ckpt

从.ckpt中加载参数并导入模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_ckpt





保存与加载数据
------------------------

保持数据到.npy文件
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_any_to_npy

从.npy文件加载数据
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_npy_to_any



文件夹/文件相关函数
------------------

判断文件存在
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: file_exists

判断文件夹存在
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: folder_exists

删除文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: del_file

删除文件夹
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: del_folder

读取文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: read_file

从文件夹中读取文件名列表
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_file_list

从文件夹中读取文件夹列表
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_folder_list

查看或建立文件夹
^^^^^^^^^^^^^^^^^^
.. autofunction:: exists_or_mkdir

下载或解压
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: maybe_download_and_extract


排序
------------

字符串按数字排序
^^^^^^^^^^^^^^^^^^^
.. autofunction:: natural_keys


可视化 npz 文件
----------------------
.. autofunction:: npz_to_W_pdf

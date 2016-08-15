.. _installation

=================
安装
=================

TensorLayer有许多需要先安装的先决条件，包括 `TensorFlow <https://www.tensorflow.org>`_ ,
numpy和matplotlib。对于GPU的支持，CUDA和cuDNN是必须的。

如果您遇到了任何麻烦，请您查看 `TensorFlow installation instructions <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_
它包含了可安装Tensorflow的一系列可选操作系统，包括 Mac OX和Linux。

如果您在运行是出现了错误，请检查 `TensorFlow installation instructions <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_
它包含了可安装Tensorflow的一系列可选操作系统，包括 Mac OX和Linux。
或者在 `hao.dong11@imperial.ac.uk <hao.dong11@imperial.ac.uk>`_ 上寻求帮助。

先决条件
===========

Python + pip
-------------

TensorLayer是基于 Python-version TensorFlow。请您首先安装python。

.. note::
    为了未来着想，我们强烈建议使用python3而不是python2来编程。

python中包括为了安装附加模块 ``pip`` 命令。此外，一个虚拟环境 <http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
通过 ``virtualenv`` 可以协助管理您的python包。

以python3为例，安装python带 ``pip`` 运行命令如下：


.. code-block:: bash

  sudo apt-get install python3
  sudo apt-get install python3-pip
  sudo pip3 install virtualenv

要构建一个需您的环境和安装matplotlib和numpy，运行下列命令：

.. code-block:: bash

  virtualenv env
  env/bin/pip install matplotlib
  env/bin/pip install numpy

检查已安装的软件包。运行下列命令：

.. code-block:: bash

  env/bin/pip list

TensorFlow
---------------

TensorFlow 网站就能看到很详细的安装说明。不过也有一些需要考虑的东西。

TensorFlow release
========================

TensorFlow
-----------


`TensorFlow <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_ 官方目前仅支持Linux和Mac OX GPU
如果您想使用GPU和Mac OX，您需要从源代码编译TensorFlow。

.. warning::
    对于ARM架构的处理器，您也可以从源代码安装TensotFlow

TensorLayer
-----------

安装TensorLayer最简单的方法如下：

.. code-block:: bash

  pip install git+https://github.com/zsdonghao/tensorlayer.git

如果您要修改或者扩展TensorLayer，您可以下载源代码，然后按如下所示安装它：

.. code-block:: bash

  cd to the root of the git tree
  pip3 install . -e

此命令将运行 ``setup.py`` 来安装TensorLayer。
``-e`` 允许您可以编辑``tensorlayer``文件中的脚本，
它能让您扩展和修改TensorLayer更加容易。

GPU支持
============

得益于NVIDA的支持，用GPU训练一个全连接的神经网络可能比用CPU训练它们要快10到20倍。
对于卷积神经网络，可能会快50倍。这要求一个支持CUDA和cuDNN的NVIDA GPU。

TensorFlow网站也教如何安装CUDA和cuDNN，请点击：
`TensorFlow: CUDA install <https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux>`_.

从NVIDA网站上安装最新版本的CUDA和cuDNN：

`CUDA install <https://developer.nvidia.com/cuda-downloads>`_

`cuDNN install <https://developer.nvidia.com/cuda-downloads>`_

在安装完毕后，请确保 ``/usr/local/cuda/bin`` 是您的 ``PATH``（使用 ``echo #PATH`` 来查看),并且 ``nvcc--version`` 工作无误。
此外要确保 ``/usr/local/cuda/lib64`` 是您的 ``LD_LIBRARY_PATH`` ，这样CUDA库才可以被发现。

如果CUDA设置正确，下面的命令应该会在终端上打印一些GPU信息：

.. code-block:: bash

  python -c "import tensorflow"

cuDNN
------------

NVIDA提供了一个对常见神经网络操作的库，特别是加快卷积神经网络(CNNs)的训练速度。
另外，在注册为开发人员之后(它需要一点时间)，它可以从NVIDA网站上得到：
`cuDNN install <https://developer.nvidia.com/cuda-downloads>`_

要安装它，复制 ``*.h`` 文件到 ``/usr/local/cuda/include`` 并且 复制``lib*`` 到
``/usr/local/cuda/lib64`` 。


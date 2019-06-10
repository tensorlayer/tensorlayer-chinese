.. _installation:

======================
安装 Installation
======================

安装 TensorFlow
=========================

.. code-block:: bash

  pip3 install tensorflow-gpu==2.0.0a0 # specific version  (YOU SHOULD INSTALL THIS ONE NOW)
  pip3 install tensorflow-gpu # GPU version
  pip3 install tensorflow # CPU version

更多TensorFlow安装信息，可在Google官网查看。TensorFlow支持Linux、MscOS和Windows下的GPU加速，需要用户自行安装CUDA和CuDNN。

安装 TensorLayer
=========================

稳定版本:

.. code-block:: bash

  pip3 install tensorlayer
  
最新版本请通过Github来安装:

.. code-block:: bash

  pip3 install git+https://github.com/tensorlayer/tensorlayer.git
  or
  pip3 install https://github.com/tensorlayer/tensorlayer/archive/master.zip

对于TensorLayer贡献者，建议从Github把整个项目clone到本地，然后把tensorlayer文件夹放到相应的项目中去。

.. code-block:: bash

  git clone https://github.com/tensorlayer/tensorlayer.git

您也可以通过源码来安装:

.. code-block:: bash

  # 首先把TensorLayer从Github下载到本地
  git clone https://github.com/tensorlayer/tensorlayer.git
  cd tensorlayer

  # 建议安装 virtualenv
  pip install virtualenv
  # 创造虚拟环境 `venv`
  virtualenv venv

  # 激活虚拟环境

  ## Linux:
  source venv/bin/activate

  ## Windows:
  venv\Scripts\activate.bat

  # 简单安装
  pip install .

  # ============= IF TENSORFLOW IS NOT ALREADY INSTALLED ============= #

  # for a machine **without** an NVIDIA GPU
  pip install -e ".[all_cpu_dev]"

  # for a machine **with** an NVIDIA GPU
  pip install -e ".[all_gpu_dev]"

如果您想使用旧版的TensorLayer 1.X:

.. code-block:: bash

  [stable version] pip install tensorlayer==1.x.x

如果您想修改旧版的TensorLayer 1.X，您也可以把整个项目下载下来，再安装

.. code-block:: bash

  cd to the root of the git tree
  pip install -e .

这个命令会根据 ``setup.py`` 来安装TensorLayer。符号 ``-e`` 表示可修改（editable），这样您可以修改 ``tensorlayer`` 文件夹中的源码，然后 ``import`` 使用之。


GPU 加速 
==========================

CUDA
----

TensorFlow 官网也提供了安装 CUDA 和 CuDNN 的教程。简单来说，请先从NVIDIA官网下载CUDA：

 - `CUDA 下载与安装 <https://developer.nvidia.com/cuda-downloads>`_


..
  make sure ``/usr/local/cuda/bin`` is in your ``PATH`` (use ``echo #PATH`` to check), and
  ``nvcc --version`` works. Also ensure ``/usr/local/cuda/lib64`` is in your
  ``LD_LIBRARY_PATH``, so the CUDA libraries can be found.

如果 CUDA 安装成功，请使用如下命令来显示GPU的信息。

.. code-block:: bash

  python -c "import tensorflow"


CuDNN
--------

除了 CUDA, NVIDIA 提供一个针对深度学习加速的库--CuDNN。您需要注册NVIDIA开发者，然后才能下载它：

 - `CuDNN 下载连接 <https://developer.nvidia.com/cudnn>`_

下载解压后，把 ``*.h`` 文件复制到 ``/usr/local/cuda/include`` 并把
``lib*`` 文件复制到 ``/usr/local/cuda/lib64``.

.. _TensorFlow: https://www.tensorflow.org/versions/master/get_started/os_setup.html
.. _GitHub: https://github.com/tensorlayer/tensorlayer
.. _TensorLayer: https://github.com/tensorlayer/tensorlayer/



Windows 用户
==============

TensorLayer是一个Python库，因此请先给您的Windows安装Python，我们建议安装Python3.5以上的版本。

`Anaconda 下载 <https://www.continuum.io/downloads>`_

GPU 支持
------------

1. 安装 Microsoft Visual Studio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
您需要先安装Microsoft Visual Studio (VS)再安装CUDA。最低的版本要求是 VS2010，我们建议安装 VS2015 以上的版本。

2. 安装 CUDA
^^^^^^^^^^^^^^^^^^^^^^^
下载并安装最新的CUDA:

`CUDA download <https://developer.nvidia.com/CUDA-downloads>`_

3. 安装 CuDNN
^^^^^^^^^^^^^^^^^^^^^^
NVIDIA CUDA® Deep Neural Network library (cuDNN) 是一个针对深度学习开发的GPU加速库。您可以在NIVIDA官网下载之：

`cuDNN download <https://developer.nvidia.com/cuDNN>`_

解压下载文件后，您会得到三个文件夹 (bin, lib, include)。然后这些文件夹里的内容需要复制到CUDA的位置。(默认安装路径是`C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0`)

安装 TensorLayer
------------------------
For TensorLayer, please refer to the steps mentioned above.

.. code-block:: bash

  pip install tensorflow        #CPU version
  pip install tensorflow-gpu    #GPU version (GPU version and CPU version just choose one)
  pip install tensorlayer       #Install tensorlayer

测试
------
.. code-block:: bash

  import tensorlayer

如果CUDA，CuDNN安装成功，您会看到如下的信息。

.. code-block:: bash

  successfully opened CUDA library cublas64_80.dll locally
  successfully opened CUDA library cuDNN64_5.dll locally
  successfully opened CUDA library cufft64_80.dll locally
  successfully opened CUDA library nvcuda.dll locally
  successfully opened CUDA library curand64_80.dll locally






问题
=======

如果您在import时遇到困难，请查看 `FQA <http://tensorlayer.readthedocs.io/en/latest/user/more.html>`_.

.. code-block:: bash

  _tkinter.TclError: no display name and no $DISPLAY environment variable


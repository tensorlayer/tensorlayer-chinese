.. _installation:

======================
安装 Installation
======================

TensorLayer 需要一些预安装库，如 `TensorFlow`_ ， numpy 和 matplotlib。
对于 GPU 加速，需要安装 CUDA 和 cuDNN。

如果你遇到麻烦，可以查看 `TensorFlow 安装手册 <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_
，它包含了在不同系统中安装 TensorFlow 的步骤。或发邮件到 `hao.dong11@imperial.ac.uk <hao.dong11@imperial.ac.uk>`_ 询问。



步骤 1 : 安装依赖库 dependencies
====================================

TensorLayer 是运行在 python 版本的 TensorFlow 之上的，所以请先安装 python。


.. note::
    着眼于未来，我们建议使用 python3 而不是 python2

Python 的 ``pip`` 可以帮助您很快地安装库，此外 `虚拟环境(Virtual environment)
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_ 如 ``virtualenv`` 可以帮助你管理 python 包。

以 python3 和 Ubuntu 为例，可如下安装 python 及 ``pip``:

.. code-block:: bash

  sudo apt-get install python3
  sudo apt-get install python3-pip
  sudo pip3 install virtualenv

接着在虚拟环境中安装 dependencies 到虚拟环境如下:
(您也可以跳过该部分，在步骤3中让 TensorLayer 自动安装 dependencies)

.. code-block:: bash

  virtualenv env
  env/bin/pip install matplotlib
  env/bin/pip install numpy
  env/bin/pip install scipy
  env/bin/pip install scikit-image

安装完后，若无报错，可以如下在命令窗口中显示列出安装好的包:

.. code-block:: bash

  env/bin/pip list


最后，你可以用虚拟环境中的 python 来运行 python 代码，如下:

.. code-block:: bash

  env/bin/python *.py




步骤 2 : TensorFlow
=========================

TensorFlow 的安装步骤在 `TensorFlow`_  官网中有非常详细的说明，不国有一些东西是需要考虑的。
如 `TensorFlow`_ 官方目前支持 Linux, Mac OX 和 Windows 系统。

.. warning::
  对于使用 ARM 架构的机器，你需要用源码来编译安装 TensorFlow。


步骤 3 : TensorLayer
=========================

最便捷安装 TensorLayer 只需要一个指令，如下:

.. code-block:: bash

  pip install git+https://github.com/zsdonghao/tensorlayer.git

不过，若你是高级玩家，你想要在 TensorLayer 的基础上拓展或修改源码，你可以从 `Github`_ 中把整个项目下载下来，
然后如下安装。

.. code-block:: bash

  cd 到项目文件
  pip install -e .

这个命令会运行 ``setup.py`` 来安装 TensorLayer。 其中， ``-e`` 代表
可编辑的（editable），因此你可以直接修改 ``tensorlayer`` 文件夹中的源代码，然后 ``import`` 该文件夹来使用修改后的 TensorLayer。



步骤 4 : GPU 加速
==========================

非常感谢 NVIDIA 的支持，在 GPU 上训练全连接神经网络比在 CPU 上训练往往要快 10~20 倍。
对于卷积神经网络，往往会快 50 倍。这需要有一个 NIVIDA 的 GPU，以及安装 CUDA 和 cuDNN。



CUDA
----

TensorFlow 官网讲了如何安装 CUDA 和 cuDNN，`TensorFlow GPU 支持 <https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux>`_。

可在 NIVIDIA 官网下载与安装最新版本的 CUDA。

 - `CUDA 下载与安装 <https://developer.nvidia.com/cuda-downloads>`_


..
  After installation, make sure ``/usr/local/cuda/bin`` is in your ``PATH`` (use ``echo #PATH`` to check), and
  ``nvcc --version`` works. Also ensure ``/usr/local/cuda/lib64`` is in your
  ``LD_LIBRARY_PATH``, so the CUDA libraries can be found.

若 CUDA 被正确地安装，下面的指令可以在命令窗口中打印出 GPU 的信息。

.. code-block:: bash

  python -c "import tensorflow"


cuDNN
--------

出了 CUDA，NVIDIA 还专门提供另一个库来加速神经网络的运算，特别是用来加速卷积神经网络。
这个库也可以从 NIVIDIA 官网中下载安装，但你要先注册为 NIVIDA 开发者（这需要一些审核时间）。
下载时，请在 Deep Learning Framework 处在 Other 中输入 TensorLayer。


最新 cuDNN 下载与安装链接：

 - `cuDNN 下载与安装 <https://developer.nvidia.com/cudnn>`_


下载后, 复制 ``*.h`` 文件到 ``/usr/local/cuda/include`` 以及复制
``lib*`` 文件到 ``/usr/local/cuda/lib64``。



困难
=======

当你 import tensorlayer 时出现如下错误，请阅读  `FQA <http://tensorlayer.readthedocs.io/en/latest/user/more.html>`_ 。

.. code-block:: bash

  _tkinter.TclError: no display name and no $DISPLAY environment variable


.. _TensorFlow: https://www.tensorflow.org/versions/master/get_started/os_setup.html
.. _GitHub: https://github.com/zsdonghao/tensorlayer
.. _TensorLayer: https://github.com/zsdonghao/tensorlayer/

API - 预训练模型
================================

TensorLayer 提供了一些预训练模型，通过这套API，您可以非常容易地使用整个或者部分网络。

.. automodule:: tensorlayer.models

.. autosummary::

    Model

    VGG16
    VGG19
    SqueezeNetV1
    MobileNetV1

    Seq2seq
    Seq2seqLuongAttention


模型基类
-------------------------

.. autoclass:: Model


VGG16
----------------------

.. autofunction:: VGG16

VGG19
----------------------

.. autofunction:: VGG19

SqueezeNetV1
----------------
.. autofunction:: SqueezeNetV1

MobileNetV1
----------------

.. autofunction:: MobileNetV1


Seq2seq
--------------------------

.. autoclass:: Seq2seq

使用Luong注意力机制的Seq2seq
-------------------------------

.. autoclass:: Seq2seqLuongAttention

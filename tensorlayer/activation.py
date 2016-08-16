#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf

def identity(x):
    """线性激活函数

    参数
    ----------
    x : Tensor输入
        input(s)
    """
    return x

def ramp(x=None, v_min=0, v_max=1, name=None):
    """斜坡激活函数

    参数
    ----------
    x : Tensor输入
        input(s)
    v_min : 浮点数
        if input(s) smaller than v_min, change inputs to v_min
    v_max : 浮点数
        if input(s) greater than v_max, change inputs to v_max
    name : 字符串或 None
        可选输入，该激活函数的名字
    """
    return tf.clip_by_value(x, clip_value_min=v_min, clip_value_max=v_max, name=name)

#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf

def identity(x):
    return x

def ramp(x=None, v_min=0, v_max=1, name=None):
    return tf.clip_by_value(x, clip_value_min=v_min, clip_value_max=v_max, name=name)

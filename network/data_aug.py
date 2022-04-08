import tensorflow as tf
import numpy as np
import math
import os
import SimpleITK as sitk
import sys
import h5py
import random
from .spatial_transformer import Dense3DSpatialTransformer
from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import manip_ops

def random_affine(img,batch_size,
                  scale_range = [0.95,1.05,0.95,1.05,0.95,1.05],#0.9,1.1
                  degree_range = [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1],#0.3
                  translaton_range = [-2,2,-2,2,-2,2],#3
                  sx=None, sy=None, sz=None):
    
    #scale param
    s_x = tf.random_uniform([],minval = scale_range[0],maxval = scale_range[1])
    s_y = tf.random_uniform([],minval = scale_range[2],maxval = scale_range[3])
    s_z = tf.random_uniform([],minval = scale_range[4],maxval = scale_range[5])    
    #rotation param
    yam = tf.random_uniform([],minval = degree_range[0],maxval = degree_range[1])
    pitch = tf.random_uniform([],minval = degree_range[2],maxval = degree_range[3])
    roll = tf.random_uniform([],minval = degree_range[4],maxval = degree_range[5])
    #translation param
    t_x = tf.random_uniform([],minval = translaton_range[0],maxval = translaton_range[1])
    t_y = tf.random_uniform([],minval = translaton_range[2],maxval = translaton_range[3])
    t_z = tf.random_uniform([],minval = translaton_range[4],maxval = translaton_range[5])

    I = tf.convert_to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    R_z = tf.convert_to_tensor([[[tf.cos(yam), -tf.sin(yam), 0.0], [tf.sin(yam), tf.cos(yam), 0.0], [0.0, 0.0, 1.0]]])
    R_y = tf.convert_to_tensor([[[tf.cos(pitch), 0.0, tf.sin(pitch)], [0.0, 1.0, 0.0], [-tf.sin(pitch), 0.0, tf.cos(pitch)]]])
    R_x = tf.convert_to_tensor([[[1.0, 0.0, 0.0], [0.0, tf.cos(roll), -tf.sin(roll)], [0.0, tf.sin(roll), tf.cos(roll)]]])

    S = tf.convert_to_tensor([[[s_x, 0.0, 0.0], [0.0, s_y, 0.0], [0.0, 0.0, s_z]]])
    b = tf.convert_to_tensor([[t_x,t_y,t_z]])
    b = tf.reshape(b, [batch_size, 3])
    #b = tf.placeholder_with_default(b, shape = [None, 3])
    A = tf.matmul(R_z,R_y)
    A = tf.matmul(A,R_x)
    A = tf.matmul(A,S)
    #A = tf.placeholder_with_default(A, shape = [None, 3, 3])
    A = tf.reshape(A, [batch_size, 3, 3])
    print(A.shape.as_list(),b.shape.as_list(),'^'*40)
    #b = tf.tile(b, [3, 1])
    #A = tf.tile(A, [3, 1,1])

    # the flow is displacement(x) = place(x) - x = (Ax + b) - x
    W = A - I
    if sx == None:
        sx, sy, sz = img.shape.as_list()[1:4]
    flow = affine_flow(W, b, sx, sy, sz)
    print(flow.shape.as_list(),'^'*40)
    return flow

def random_intensity(img):
    beta = tf.random_uniform([],minval = 0.5,maxval = 1.5)
    #mean, var = tf.nn.moments(img,axes=[1,2,3,4])
    Max = tf.reduce_max(img)
    img = img/Max
    img = tf.pow(img ,beta)
    img = img*Max
    return img

def affine_flow(W, b, len1, len2, len3):
    b = tf.reshape(b, [-1, 1, 1, 1, 3])
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr * wx + yr * wy) + (zr * wz + b)

def affine_intensity(img,sx=None, sy=None, sz=None):
    affine_img,flow = random_affine(img,sx=sx, sy=sy, sz=sy)
    result = random_intensity(affine_img)
    return result, flow

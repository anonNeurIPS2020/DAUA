# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from util import *

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def merge_output_nonlocal(inputs):
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    x, y = inputs
    y = K.softmax(y,axis=1)
    innerProduct = K.sum(x*y,axis=1) 
#     innerProduct = K.mean(x*y,axis=1)
    return innerProduct


        
def get_grad_norm(grad_r):
    grad_norm = 0.0
    for i_grad_r,e_grad_r in enumerate(grad_r):
        if e_grad_r is None:
            break
        else:
#             print 'e_grad_r',e_grad_r
#             print 'e_grad_r.get_shape()',e_grad_r.get_shape()
            grad_norm += K.sum(K.square(e_grad_r))
    grad_norm = K.maximum(0.0*grad_norm + K.epsilon(),grad_norm)
    grad_norm = K.sqrt(grad_norm)
    return grad_norm

def get_grad_prod(grad_1,grad_2):
    grad_prod = 0.0
    for i in range(len(grad_1)):
        if grad_1[i] is None or grad_2[i] is None :
            break
        else:
            grad_prod += K.sum(grad_1[i]*grad_2[i])
    
    return grad_prod

def momentum_update(model_1,model_2,alpha=1.):
    weights_1 = model_1.get_weights()
    weights_2 = model_2.get_weights()
    weights = []
    for i in range(len(weights_1)):
        if weights_1[i] is None :
            w = None
            continue
        elif weights_2[i] is None :
            w = weights_1[i] 
            continue
        else:
            w = weights_1[i] * alpha + weights_2[i] * (1.-alpha)
        weights.append(w)
    model_1.set_weights(weights)
    
    return model_1
  
def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    return

def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten
    
def outer_product(inputs):
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    x, y = inputs
    batchSize = K.shape(x)[0]
    outerProduct = K.expand_dims(x,axis=2) * K.expand_dims(y,axis=1)
    outerProduct = K.reshape(outerProduct, (batchSize, -1))
    # returns a flattened batch-wise set of tensors
    return outerProduct

 

def merge_output_multimodel(inputs_list):
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    MODEL_NUM = len(inputs_list) - 1
    y = inputs_list[-1]
    
    batchSize = K.shape(y)[0]
#     for i in range(MODEL_NUM):
#         inputs_list[i] = K.expand_dims(inputs_list[i],axis=1)
    x = K.concatenate([K.expand_dims(inputs_list[i],axis=1) for i in range(MODEL_NUM)],axis=1)
    
    outerProduct = K.sum(x * K.expand_dims(y,axis=2),axis=1)
#     outerProduct = K.reshape(outerProduct, (batchSize, -1))
    # returns a flattened batch-wise set of tensors
    return outerProduct

def prob2extreme(prob):
    scaled_prob = 2 * prob - 1
    extremed_prob=np.sin(scaled_prob*np.pi/2)
    return (extremed_prob+1)/2
 


def my_dense(x,W,b):
    return np.dot(x,W)+b[np.newaxis,:]
def my_sigmoid(x,alpha=1):
    return 1./(1.+np.exp(-alpha*x))
def my_swish(x):
    return x*my_sigmoid(x)
def my_undropout(x,p=0.5):
    return x*p
def my_batchnorm(x,gamma,beta,moving_mean,moving_variance):
    inv = 1./np.sqrt(moving_variance + 1e-3)
    inv = inv * gamma
    return x * inv + beta - moving_mean * inv
def my_tanh(x):
    return my_sigmoid(x,alpha=2) * 2. - 1.

def my_zscore(X):
    mean_X = np.mean(X, 0)
    std_X = np.std(X, 0)
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X, mean_X, std_X


def my_zscore_test(X, mean_X, std_X):
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X

def my_dropout(X,p=0.5):
#     print('p',p)
    win = np.random.rand(X.shape[0],X.shape[1])
    win = win >= p
    X *= win
    return X

def scheduler(model,lr):
    K.set_value(model.optimizer.lr, lr)
    return K.get_value(model.optimizer.lr)

def scheduler_fixdecay(model,ratio):
    lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, lr*ratio)
    return K.get_value(model.optimizer.lr)

def layer_select_col(x,feature_dim=128):
    return x[:,:feature_dim]
 
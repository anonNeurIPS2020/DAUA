# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
 

from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU,ELU,LeakyReLU

from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from scipy.io import loadmat
 

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy.optimize import minimize_scalar

import pandas as pd

from keras import regularizers
from keras import initializers

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.engine import *
 
    
def swish(x):
    y = Activation('sigmoid')(x)
    return multiply([x,y])
 
def ada_eye(mat):
    col = K.tf.reduce_sum(mat, 1)
    col = K.tf.ones_like(col)
    return K.tf.diag(col)

def orth_reg(W):
    WT = K.transpose(W)
    X = K.dot(WT,W)
    X = X *(1.- ada_eye(X))
    return 1e-8 * K.sum(X*X)

 
def get_generative(input_dim=32, dense_dim=200, out_dim=50,activation='tanh', lr=1e-3,activity_l1 = 0.0,flag_orth_reg=False,flag_orth_init=False,flag_SN=False,kernel_l2=0.0,flag_he_init=False,flag_BN=False,depth=1,flag_resnet=False,flag_dropout=False):
    G_in = Input([input_dim])
 
    if flag_orth_reg:
        regu = orth_reg
    else:
        regu = regularizers.l2(kernel_l2)
    
    activity_regu = regularizers.l1(activity_l1)
         
        
    if flag_orth_init:
        init = initializers.Orthogonal()
    elif flag_he_init:
        init = 'he_normal'
    else:
        init = 'glorot_uniform'
        
    if flag_SN:
        Dense_final = DenseSN
    else:
        Dense_final = Dense
   
    x = G_in
    
    for i in range(depth):
        
        if flag_resnet:
            if i >= 1 and i < depth - 1:
                y = x
                
        if flag_dropout:
            x = Dropout(0.5)(x)
    
        x = Dense_final(out_dim, activation='linear',activity_regularizer=activity_regu,kernel_regularizer=regu,kernel_initializer=init)(x)  

        if flag_BN:
            x = BatchNormalization()(x)
            
        if depth > 1 and i < depth - 1:
            x = swish(x)
            
        else:

            if activation == 'swish':
                x = swish(x)
            else:
                x = Activation(activation)(x)
            
        if flag_resnet:
            if i >= 1 and i < depth - 1:
                x = add([y,x])  
 
        
    G_out = x
    G = Model(G_in, G_out)
    opt = Adam(lr=lr,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    G.compile(loss='mse', optimizer=opt)   
    return G, G_out
 
def get_discriminative(input_dim=32,hidden_dim_list=None, out_dim=50,activation='sigmoid', lr=1e-3,kernel_l1 = 0.0,kernel_l2=0.0,flag_norm1=False,flag_orth_reg=False,flag_orth_init=False,flag_he_init=False,flag_BN=False,depth=1,flag_resnet=False,flag_finalBN=True):
    D_in = Input([input_dim])
    if kernel_l1 > 0.0:
        regu = regularizers.l1(kernel_l1)
    else:
        regu = regularizers.l2(kernel_l2)
    
        
    if flag_orth_reg:
        regu = orth_reg
 
        
        
        
    if flag_orth_init:
        init = initializers.Orthogonal()
    elif flag_he_init:
        init = 'he_normal'
    else:
        init = 'glorot_uniform'
        
    x = D_in
    
    for i in range(depth):
        
        if flag_resnet:
            if i >= 1 and i < depth - 1:
                y = x
                
        if hidden_dim_list is not None:
            x = Dense(hidden_dim_list[i], activation='linear',kernel_regularizer=regu,kernel_initializer=init)(x)
        else:
            x = Dense(out_dim, activation='linear',kernel_regularizer=regu,kernel_initializer=init)(x)
    
        if flag_BN:
            if i == depth - 1:
                if flag_finalBN:
                    x = BatchNormalization()(x)
            else:
                x = BatchNormalization()(x)

        if flag_norm1:
            x = Lambda(lambda x:K.l2_normalize(x,axis=1))(x)   
            
        if depth > 1 and i < depth - 1:
            x = swish(x)
            
        else:

            if activation == 'swish':
                x = swish(x)
            else:
                x = Activation(activation)(x)
            
        if flag_resnet:
            if i >= 1 and i < depth - 1:
                x = add([y,x])  
         
    D_out = x
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr,beta_1=0.5,beta_2=0.999)
    D.compile(loss='mse', optimizer=dopt)   
    return D, D_out
 
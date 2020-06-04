# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
 

from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Dense
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
import resnet2

from keras import regularizers


from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers
import tensorflow as tf
from keras.engine import *

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations 
     
def swish(x):
    y = Activation('sigmoid')(x)
    return multiply([x,y])


def outer_swish(x):
    
    y=Lambda(outer_swish_sub)(x)
    
    return y

def outer_swish_sub(x):
    y = K.sigmoid(x)
    outerProduct = K.expand_dims(x,axis=2) * K.expand_dims(y,axis=1)
    return K.logsumexp(outerProduct,axis=1)

 
def input_model_fc(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")  
    x = Dense(hidden_dim, activation='linear')(inputs)  
    x = BatchNormalization()(x)
    x0 = swish(x)
#     flat = x0
    flat = Dropout(0.5)(x0)
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten
 
def input_model_simpleCNN(hidden_dim,input_shape): 
    inputs = Input(input_shape, name="input")
    conv1 = Conv2D(32, (3, 3), activation="relu", name="conv1")(inputs)
    conv2 = Conv2D(32, (3, 3), activation="relu", name="conv2")(conv1)
    pool = MaxPooling2D((2, 2), name="pool1")(conv2)
    drop = Dropout(0.25, name="drop")(pool)
    flat = Flatten(name="flat")(drop)
    model_input = Model(inputs, flat)
    
     
    shape_before_flatten = drop.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_flatten

 

def ada_eye(mat):
    col = K.tf.reduce_sum(mat, 1)
    col = K.tf.ones_like(col)
    return K.tf.diag(col)

def orth_reg(W):
    WT = K.transpose(W)
    X = K.dot(WT,W)
    X = X *(1.- ada_eye(X))
    return 1e-3 * K.sum(X*X)
 

 
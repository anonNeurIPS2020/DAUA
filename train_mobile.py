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

from model import reader_tensor
from model import reader_vector

import tensorflow as tf


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
 
from data.gendata_sensortouch_grouped_classes import gen_raw_data
 
from nets.nets import input_model_fc as input_model


from nets.GD_basic import get_generative,get_discriminative

from util.util import label2uniqueID,split_test_as_valid,argmin_mean_FAR_FRR,auc_MTL,FAR_score,FRR_score,evaluate_result_valid,evaluate_result_test,mkdir
from util.util_keras import set_trainability
from losses.losses import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
 
if K.backend() == "tensorflow":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
 
    session = tf.Session(config=config)
    K.set_session(session)
    
def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] 
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten

def np_output(y_pred):
    k = 2
    y_shape_1 = y_pred.shape[1]
#     print 'y_shape_1',y_shape_1
    c = (y_shape_1-k)/k
#     print 'c',c
    y_prior = y_pred[:,-k:]
    y_out = 0.0
    for r in range(k):
        y_out += y_pred[:,r*c:(r+1)*c]*np.expand_dims(y_prior[:,r],axis=1)
#     r=0
#     y_out = y_pred[:,r*c:(r+1)*c]
        
    return y_out

COMP_NUM = 2
flag_stop_gradient = True
flag_prior_mse = False
flag_posterior_max_entropy = False
flag_anti_max_entropy = False
    
def moe_loss(y_true, y_pred):
    k = COMP_NUM
    y_shape_1 = my_get_shape(y_pred)
    c = (y_shape_1-k)/k
    y_prior = y_pred[:,-k:]
    y_ll_list = []
    y_prior_logit_list = []
    y_posterior_logit_list = []
    for r in range(k):
        y_pred_r = y_pred[:,r*c:(r+1)*c]
        y_ll_r = K.sum(y_true * K.log(y_pred_r+K.epsilon()),axis=1,keepdims=True)
        y_ll_list.append(y_ll_r)
        
        y_prior_logit_r = K.expand_dims(K.log(y_prior[:,r]+K.epsilon()),axis=1)
        y_prior_logit_list.append(y_prior_logit_r)
        
        y_posterior_logit_r = y_ll_r + y_prior_logit_r
        y_posterior_logit_list.append(y_posterior_logit_r)
    y_posterior_logit = K.concatenate(y_posterior_logit_list,axis=1)
    
    posterior = K.softmax(y_posterior_logit,axis=1)
    posterior_nograd = K.stop_gradient(posterior)
 
    
    loss_pred = 0.0
    for r in range(k):
        if flag_stop_gradient:
            loss_pred += -K.expand_dims(posterior_nograd[:,r],axis=1) * y_ll_list[r]#/K.sum(posterior[:,r]+1.)
        else:
            loss_pred += -K.expand_dims(posterior[:,r],axis=1) * y_ll_list[r]#/K.sum(posterior[:,r]+1.)
 
    loss_prior = 0.0
    for r in range(k):
        if flag_stop_gradient:
            loss_prior += -K.expand_dims(posterior_nograd[:,r],axis=1) * y_prior_logit_list[r]
        else:
            loss_prior += -K.expand_dims(posterior[:,r],axis=1) * y_prior_logit_list[r]
        
#     loss_posterior = ori_prob_categorical_crossentropy(posterior,posterior)
    if flag_posterior_max_entropy:
        loss_posterior = K.sum(posterior*K.log(posterior+K.epsilon()),axis=1,keepdims=True)
    else:
        loss_posterior = ori_prob_categorical_crossentropy(posterior,posterior)
        
    posterior_mean = K.mean(posterior,axis=0)
    prior_mean = K.mean(y_prior,axis=0)
    
    loss_mean_posterior_prior = K.sum(posterior_mean*K.log(prior_mean +K.epsilon()))
    
    
    return K.mean(loss_pred + 1.0 * loss_prior + 0.0*loss_posterior + 1.0 *loss_mean_posterior_prior
                 +1.0*ori_prob_categorical_crossentropy(y_prior,y_prior))

#     return ori_prob_categorical_crossentropy(y_prior,y_prior)

def moe_mse_loss_pos(y_true, y_pred):
    k = COMP_NUM
    y_shape_1 = my_get_shape(y_pred)
    c = (y_shape_1-k)/k
    y_prior = y_pred[:,-k:]
    y_ll_list = []
    y_prior_logit_list = []
    y_posterior_logit_list = []
    y_pred_mse_list = []
    y_prior_mse_list = []
    for r in range(k):
        y_pred_r = y_pred[:,r*c:(r+1)*c]
        y_ll_r = K.sum(y_true * K.log(y_pred_r+K.epsilon()),axis=1,keepdims=True)
        y_ll_list.append(y_ll_r)
        
        y_pred_mse_r = K.sum(K.square(y_true - y_pred_r),axis=1,keepdims=True)
        y_pred_mse_list.append(y_pred_mse_r)
        
        y_prior_logit_r = K.expand_dims(K.log(y_prior[:,r]+K.epsilon()),axis=1)
        y_prior_logit_list.append(y_prior_logit_r)
        
        y_posterior_logit_r = y_ll_r + y_prior_logit_r
        y_posterior_logit_list.append(y_posterior_logit_r)
    y_posterior_logit = K.concatenate(y_posterior_logit_list,axis=1)
    
    posterior = K.softmax(y_posterior_logit,axis=1)
    posterior_nograd = K.stop_gradient(posterior)
 
    
    loss_pred = 0.0
    for r in range(k):
        if flag_stop_gradient:
            loss_pred += -K.expand_dims(posterior_nograd[:,r],axis=1) * y_ll_list[r]#/K.sum(posterior[:,r]+1.)
        else:
            loss_pred += -K.expand_dims(posterior[:,r],axis=1) * y_ll_list[r]#/K.sum(posterior[:,r]+1.)
 

    if flag_prior_mse:
        loss_prior = K.sum(K.square(posterior-y_prior),axis=1,keepdims=True)
    else:
        loss_prior = 0.0
        for r in range(k):
            if flag_stop_gradient:
                loss_prior += -K.expand_dims(posterior_nograd[:,r],axis=1) * y_prior_logit_list[r]
            else:
                loss_prior += -K.expand_dims(posterior[:,r],axis=1) * y_prior_logit_list[r]

    if flag_posterior_max_entropy:
        loss_posterior = K.sum(posterior*K.log(posterior+K.epsilon()),axis=1,keepdims=True)
    else:
        loss_posterior = ori_prob_categorical_crossentropy(posterior,posterior)
        
    posterior_mean = K.mean(posterior,axis=0)
    prior_mean = K.mean(y_prior,axis=0)
    
    loss_mean_posterior_prior = K.sum(posterior_mean*K.log(prior_mean +K.epsilon()))
    
    return K.mean(loss_pred + 1.0 * loss_prior + 0.0*loss_posterior+ 1.0 *loss_mean_posterior_prior + 
                  1.0 *ori_prob_categorical_crossentropy(y_prior,y_prior))

#     return ori_prob_categorical_crossentropy(y_prior,y_prior)

def moe_mse_loss_neg(y_true, y_pred):
    k = COMP_NUM
    y_shape_1 = my_get_shape(y_pred)
    c = (y_shape_1-k)/k
    y_prior = y_pred[:,-k:]
    y_ll_list = []
    y_prior_logit_list = []
    y_posterior_logit_list = []
    y_pred_mse_list = []
    y_prior_mse_list = []
    y_pred_list = []
    for r in range(k):
        y_pred_r = y_pred[:,r*c:(r+1)*c]
        y_ll_r = K.sum(y_true * K.log(y_pred_r+K.epsilon()),axis=1,keepdims=True)
        y_ll_list.append(y_ll_r)
        y_pred_list.append(y_pred_r)
        
        y_pred_mse_r = K.sum(K.square(y_true - y_pred_r),axis=1,keepdims=True)
        y_pred_mse_list.append(y_pred_mse_r)
        
        y_prior_logit_r = K.expand_dims(K.log(y_prior[:,r]+K.epsilon()),axis=1)
        y_prior_logit_list.append(y_prior_logit_r)
        
        y_posterior_logit_r = y_ll_r + y_prior_logit_r
        y_posterior_logit_list.append(y_posterior_logit_r)
    y_posterior_logit = K.concatenate(y_posterior_logit_list,axis=1)
    
    
    posterior = K.softmax(y_posterior_logit,axis=1)
    posterior_nograd = K.stop_gradient(posterior)
    
    if flag_posterior_max_entropy:
        loss_posterior = K.sum(posterior*K.log(posterior+K.epsilon()),axis=1,keepdims=True)
    else:
        loss_posterior = ori_prob_categorical_crossentropy(posterior,posterior)
    
#     loss_posterior = ori_prob_categorical_crossentropy(posterior,posterior)
    
    if flag_anti_max_entropy:
        loss_posterior_anti = max_entropy(posterior,posterior)
    else:
        loss_posterior_anti = K.sum(K.square(posterior-1.),axis=1,keepdims=True)
     
    
    loss_pred = 0.0
    for r in range(k):
        if flag_stop_gradient:
#         loss_pred += K.expand_dims(posterior[:,r],axis=1)  * y_pred_mse_list[r]#/K.sum(posterior[:,r]+1.)
            loss_pred += -  y_ll_list[r]#/K.sum(posterior[:,r]+1.)
        else:
            loss_pred += -  y_ll_list[r]#/K.sum(posterior[:,r]+1.)
        
    if flag_prior_mse:
        loss_prior = K.sum(K.square(posterior-y_prior),axis=1,keepdims=True)
    else:
        loss_prior = 0.0
        for r in range(k):
            if flag_stop_gradient:
                loss_prior += -K.expand_dims(posterior_nograd[:,r],axis=1) * y_prior_logit_list[r]
            else:
                loss_prior += -K.expand_dims(posterior[:,r],axis=1) * y_prior_logit_list[r]
    
    if flag_anti_max_entropy:
        loss_prior_anti = max_entropy(y_prior,y_prior)
    else:
        loss_prior_anti = K.mean(K.square(y_prior-1.),axis=1,keepdims=True)
    
    posterior_mean = K.mean(posterior,axis=0)
    prior_mean = K.mean(y_prior,axis=0)
    
    loss_mean_posterior_prior = K.sum(posterior_mean*K.log(prior_mean +K.epsilon()))
     
    
    loss_diff_pred = K.mean(K.abs(y_pred_list[0]-y_pred_list[1]))
#     for r in range(k):
#         loss_posterior += K.square(posterior[:,r]-1./k)
    
    
    return K.mean( 0. * loss_pred + 0. *loss_prior + 0.* loss_posterior 
                  +  0.0*loss_posterior_anti+10.0*loss_prior_anti+ 0. *loss_mean_posterior_prior
                 + 0.0 * loss_diff_pred)


def moe_mse_loss_neg2(y_true, y_pred):
    k = COMP_NUM
    y_shape_1 = my_get_shape(y_pred)
    c = (y_shape_1-k)/k
    y_prior = y_pred[:,-k:]
    y_ll_list = []
    y_prior_logit_list = []
    y_posterior_logit_list = []
    y_pred_mse_list = []
    y_prior_mse_list = []
    y_pred_list = []
    for r in range(k):
        y_pred_r = y_pred[:,r*c:(r+1)*c]
        y_ll_r = K.sum(y_true * K.log(y_pred_r+K.epsilon()),axis=1,keepdims=True)
        y_ll_list.append(y_ll_r)
        y_pred_list.append(y_pred_r)
        
        y_pred_mse_r = K.sum(K.square(y_true - y_pred_r),axis=1,keepdims=True)
        y_pred_mse_list.append(y_pred_mse_r)
        
        y_prior_logit_r = K.expand_dims(K.log(y_prior[:,r]+K.epsilon()),axis=1)
        y_prior_logit_list.append(y_prior_logit_r)
        
        y_posterior_logit_r = y_ll_r + y_prior_logit_r
        y_posterior_logit_list.append(y_posterior_logit_r)
    y_posterior_logit = K.concatenate(y_posterior_logit_list,axis=1)
    
    
    posterior = K.softmax(y_posterior_logit,axis=1)
    posterior_nograd = K.stop_gradient(posterior)
    
    if flag_posterior_max_entropy:
        loss_posterior = K.sum(posterior*K.log(posterior+K.epsilon()),axis=1,keepdims=True)
    else:
        loss_posterior = ori_prob_categorical_crossentropy(posterior,posterior)
    
#     loss_posterior = ori_prob_categorical_crossentropy(posterior,posterior)
    
    if flag_anti_max_entropy:
        loss_posterior_anti = max_entropy(posterior,posterior)
    else:
        loss_posterior_anti = K.sum(K.square(posterior-1.),axis=1,keepdims=True)
     
    
    loss_pred = 0.0
    for r in range(k):
        if flag_stop_gradient:
#         loss_pred += K.expand_dims(posterior[:,r],axis=1)  * y_pred_mse_list[r]#/K.sum(posterior[:,r]+1.)
            loss_pred += -  y_ll_list[r]#/K.sum(posterior[:,r]+1.)
        else:
            loss_pred += -  y_ll_list[r]#/K.sum(posterior[:,r]+1.)
        
    if flag_prior_mse:
        loss_prior = K.sum(K.square(posterior-y_prior),axis=1,keepdims=True)
    else:
        loss_prior = 0.0
        for r in range(k):
            if flag_stop_gradient:
                loss_prior += -K.expand_dims(posterior_nograd[:,r],axis=1) * y_prior_logit_list[r]
            else:
                loss_prior += -K.expand_dims(posterior[:,r],axis=1) * y_prior_logit_list[r]
    
    if flag_anti_max_entropy:
        loss_prior_anti = max_entropy(y_prior,y_prior)
    else:
        loss_prior_anti = K.mean(K.square(y_prior-1.),axis=1,keepdims=True)
    
    posterior_mean = K.mean(posterior,axis=0)
    prior_mean = K.mean(y_prior,axis=0)
    
    loss_mean_posterior_prior = K.sum(posterior_mean*K.log(prior_mean +K.epsilon()))
     
    
    loss_diff_pred = K.mean(K.abs(y_pred_list[0]-y_pred_list[1]))
#     for r in range(k):
#         loss_posterior += K.square(posterior[:,r]-1./k)
    
    
    return K.mean( 0. * loss_pred + 0. *loss_prior + 0.* loss_posterior 
                  +  10.0*loss_posterior_anti+10.0*loss_prior_anti+ 0. *loss_mean_posterior_prior
                 + 1.0 * loss_diff_pred)

#     return loss_prior_anti
 
def make_gan(inputs, G, D, G_trainable,D_trainable):
    set_trainability(G, G_trainable)
    set_trainability(D, D_trainable)
    x = G(inputs)
    output = D(x)
    return output
 
def make_gan_phase_1_gen_hidden_feature(inputs, G_in, G_set):
    output_set = []
    ATTR_NUM = len(G_set)
    for i in range(ATTR_NUM):
        x = G_set[i](G_in)
        output_set.append(x)
    feats = concatenate(output_set)
    GAN = Model(inputs, feats)
    return GAN, feats
 
def make_gan_phase_1_task(inputs, GAN_in, G_set, D_set, loss,opt,loss_weights,CLASS_NUM):
    output_set = []
    ATTR_NUM = len(G_set)
    
    G_trainable = True
    D_trainable = True
    for i in range(ATTR_NUM):  
        if i > 0:
            output_set_sub = []
            for r in range(CLASS_NUM[1]):
                output_ii = make_gan(GAN_in, G_set[i], D_set[i][i][r], G_trainable,D_trainable)
                output_set_sub.append(output_ii)
            output_ii = make_gan(GAN_in, G_set[i], D_set[i][i][-1], G_trainable,D_trainable)
            output_set_sub.append(output_ii)
            output_ii = concatenate(output_set_sub)
        else:
            output_ii = make_gan(GAN_in, G_set[i], D_set[i][i], G_trainable,D_trainable)
        output_set.append(output_ii)
    GAN = Model(inputs, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set
 
def make_gan_phase_1_domain_pos(inputs, GAN_in, G_set, D_set, loss,opt,loss_weights,CLASS_NUM):
    output_set = []
    ATTR_NUM = len(G_set)
    G_trainable = False  
    for i in range(ATTR_NUM):
        for j in range(ATTR_NUM):
            if i == j:
                D_trainable = False  
            else:
                D_trainable = True  
            if j > 0 :
                output_set_sub = []
                for r in range(CLASS_NUM[1]):
                    output_ij = make_gan(GAN_in, G_set[i], D_set[i][j][r], G_trainable,D_trainable)
                    output_set_sub.append(output_ij)
                output_ij = make_gan(GAN_in, G_set[i], D_set[i][j][-1], G_trainable,D_trainable)
                output_set_sub.append(output_ij)
                output_ij = concatenate(output_set_sub)
            else:      
                output_ij = make_gan(GAN_in, G_set[i], D_set[i][j], G_trainable,D_trainable)
            output_set.append(output_ij)
    GAN = Model(inputs, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set
 
def make_gan_phase_1_domain_neg(inputs, GAN_in, G_set, D_set, loss,opt,loss_weights,CLASS_NUM):
    output_set = []
    ATTR_NUM = len(G_set)
    D_trainable = False  
    for i in range(ATTR_NUM):
        for j in range(ATTR_NUM):
            if i == j:
                G_trainable = False  
            else:
                G_trainable = True  
            if j > 0:
                output_set_sub = []
                for r in range(CLASS_NUM[1]):
                    output_ij = make_gan(GAN_in, G_set[i], D_set[i][j][r], G_trainable,D_trainable)
                    output_set_sub.append(output_ij)
                output_ij = make_gan(GAN_in, G_set[i], D_set[i][j][-1], G_trainable,D_trainable)
                output_set_sub.append(output_ij)
                output_ij = concatenate(output_set_sub)
            else:      
                output_ij = make_gan(GAN_in, G_set[i], D_set[i][j], G_trainable,D_trainable)
            output_set.append(output_ij)
    GAN = Model(inputs, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set
  
def build_model(ATTR_NUM,CLASS_NUM,feature_dim,hidden_dim,input_shape,lambda_mat):
     
    model_input, inputs, _, shared_dim = input_model(hidden_dim,input_shape)
     
    G_set_phase_1 = []
    D_set_phase_1 = []   
    for i in range(ATTR_NUM):
        G,_ = get_generative(input_dim=shared_dim, out_dim=feature_dim,flag_orth_init=True,flag_he_init=False,
                             flag_orth_reg=False)
        G_set_phase_1.append(G)
        D_set_sub = []
        for j in range(ATTR_NUM):
            if i == j:
                activation = 'softmax'  
            else:
                activation = 'softmax'  
            if j > 0:
                D_set_sub_sub = []
                for r in range(CLASS_NUM[1]):
                    D,_ = get_discriminative(input_dim=feature_dim, out_dim=CLASS_NUM[0],activation=activation,
                                            kernel_l1=0,flag_orth_init=False,flag_he_init=False)
                    D_set_sub_sub.append(D)
                D,_ = get_discriminative(input_dim=feature_dim, out_dim=CLASS_NUM[j],activation=activation,
                                        kernel_l1=0,flag_orth_init=False,flag_he_init=False)
                D_set_sub_sub.append(D)
                D_set_sub.append(D_set_sub_sub)
            else:
                D,_ = get_discriminative(input_dim=feature_dim, out_dim=CLASS_NUM[j],activation=activation,
                                        kernel_l1=0,flag_orth_init=False,flag_he_init=False)
                D_set_sub.append(D)
        D_set_phase_1.append(D_set_sub)
        
 
    opt_gan = Adam(lr=0.0002,beta_1=0.5,beta_2=0.999)
    opt = Adam(lr=1e-3)
    
     
    loss_weights = [1.]
    loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 
    
    set_trainability(model_input, True) 
  
    feats = model_input(inputs)
    loss = ['categorical_crossentropy',moe_loss]
    GAN_phase_1_task,_ = make_gan_phase_1_task(inputs, feats, G_set_phase_1, D_set_phase_1, loss,opt,loss_weights,CLASS_NUM)


    
    for i in range(ATTR_NUM):
        loss_weights = [1.]
        loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 
        for j in range(ATTR_NUM):
            if i!=j:
#                 print 'i,j,lambda_mat[j,i]',i,j,lambda_mat[j,i]
                loss_weights[j] = loss_weights[j] * lambda_mat[j,i]
        if i == 0:
            loss_w = loss_weights
        else:
            loss_w.extend(loss_weights)
    loss_weights = loss_w

    set_trainability(model_input, False) 
    
    feats = model_input(inputs)
 
    loss = ['mse',moe_mse_loss_pos,'mse',moe_mse_loss_pos]
#     for i in range(ATTR_NUM):
#         loss.extend(['mse',ori_prob_categorical_crossentropy])
    GAN_phase_1_domain_pos,_ = make_gan_phase_1_domain_pos(inputs, feats, G_set_phase_1, D_set_phase_1, loss,opt_gan,loss_weights,CLASS_NUM)#å…¶ä¸­'mse'å¯¹åº”LSGAN
#     GAN_phase_1_domain_pos.summary() 



    for i in range(ATTR_NUM):
        if i ==0:
            loss_weights = [0.]
            loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 
        else:
            loss_weights = [0.]
            loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 
        for j in range(ATTR_NUM):
            if i!=j:
#                 print 'i,j,lambda_mat[j,i]',i,j,lambda_mat[j,i]
                loss_weights[j] = loss_weights[j] * lambda_mat[j,i]
        if i == 0:
            loss_w = loss_weights
        else:
            loss_w.extend(loss_weights)
    loss_weights = loss_w
    
    
    set_trainability(model_input, True) 
    
    feats = model_input(inputs)
    
#     loss = []
#     for i in range(ATTR_NUM):
#         loss_sub = ['mse']
#         loss_sub.extend([moe_mse_loss_neg for _ in range(ATTR_NUM-1)])
#         loss.extend(loss_sub)
   
    loss = ['mse',moe_mse_loss_neg,'mse',moe_mse_loss_neg]
#     loss = [moe_mse_loss_neg,my_mean_squared_error,moe_mse_loss_neg_branch2_simple,my_mean_squared_error]
    GAN_phase_1_domain_neg,_ = make_gan_phase_1_domain_neg(inputs, feats, G_set_phase_1, D_set_phase_1, loss,opt_gan,loss_weights,CLASS_NUM)#å…¶ä¸­'mse'å¯¹åº”LSGAN
#     GAN_phase_1_domain_neg.summary() 


    for i in range(ATTR_NUM):
        loss_weights = [1.]
        loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 
        for j in range(ATTR_NUM):
            if i!=j:
#                 print 'i,j,lambda_mat[j,i]',i,j,lambda_mat[j,i]
                loss_weights[j] = loss_weights[j] * lambda_mat[j,i]
        if i == 0:
            loss_w = loss_weights
        else:
            loss_w.extend(loss_weights)
    loss_weights = loss_w

    set_trainability(model_input, True) 
    feats = model_input(inputs)
     
 
    loss = ['mse',moe_mse_loss_neg2,'mse',moe_mse_loss_neg2]
 
    GAN_phase_1_domain_neg2,_ = make_gan_phase_1_domain_neg(inputs, feats, G_set_phase_1, D_set_phase_1, loss,opt_gan,loss_weights,CLASS_NUM)#å…¶ä¸­'mse'å¯¹åº”LSGAN
 


 
    Model_gen_hidden_feature,_=make_gan_phase_1_gen_hidden_feature(inputs, feats, G_set_phase_1)

    return GAN_phase_1_task,GAN_phase_1_domain_pos,GAN_phase_1_domain_neg,GAN_phase_1_domain_neg2,Model_gen_hidden_feature
           
def train():   
    
     
    #è®¾ç½®å‚æ•°
    hidden_dim = 128*3 
    feature_dim = 128 
    batch_size = 512
    n_step = 100000000
    save_name = 'final' 
    
    
    train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape,data_name,lambda_mat,prior_list = gen_raw_data()
    
    idx_valid,idx_test = split_test_as_valid(test_y_c)
    val_X = test_X[idx_valid]
    val_y_a = test_y_a[idx_valid]
    val_y_c = test_y_c[idx_valid]

    test_X = test_X[idx_test]
    test_y_a = test_y_a[idx_test]
    test_y_c = test_y_c[idx_test]

    ATTR_NUM = 2
    CLASS_NUM = [CLASS_NUM[0],2]
    lambda_mat = 1 - np.eye(ATTR_NUM)
    prior_list = [prior_list[0],0.5*np.ones(2),0.1*np.ones(10)]

  

    
    print 'start building model'
 
    GAN_phase_1_task,GAN_phase_1_domain_pos,GAN_phase_1_domain_neg,GAN_phase_1_domain_neg2,Model_gen_hidden_feature = build_model(ATTR_NUM,CLASS_NUM,feature_dim,hidden_dim,input_shape,lambda_mat)   
    
    
    
     
    if np.size(input_shape) == 1:
        data_reader_train = reader_vector.Reader(train_X,train_y_a, batch_size=batch_size)  
    elif np.size(input_shape) == 3:
        data_reader_train = reader_tensor.Reader(train_X,train_y_a, batch_size=batch_size)  
    
    print 'start training'
 
    for i_step in range(n_step):
 
        x_batch, y_batch  = data_reader_train.iterate_batch()
         
 
        y_batch_task = [to_categorical(y_batch[:, 0], CLASS_NUM[0]) for _ in range(ATTR_NUM)]
 
        for _ in range(5):
            GAN_phase_1_task.train_on_batch(x_batch,y_batch_task)
         
        y_batch_domain = []
        y_batch_domain_inv = []
        for i in range(ATTR_NUM):
            for j in range(ATTR_NUM):
                yy = y_batch_task[j]
                yyy = yy.copy()
                y_batch_domain.append(yyy)
                if i == j:
                    y_batch_domain_inv.append(yyy)  
                else:
                     if j > 0:
                        zzz = y_batch_task[0]
                     else:
                        prior_vec = prior_list[j].copy()
    #                     print prior_vec
                        prior_vec = np.ones((yyy.shape[0],1))*prior_vec[np.newaxis,:]
                        prior_vec[yyy==1] = 0
                        zzz = prior_vec/np.sum(prior_vec,axis=1)[:,np.newaxis]
                        zzz = zzz * CLASS_NUM[j]
#                         zzz = np.hstack([yyy,zzz])
                     y_batch_domain_inv.append(zzz) 
                    
                    
 
        for _ in range(5):
            GAN_phase_1_domain_pos.train_on_batch(x_batch,y_batch_domain)
        
 
        for _ in range(5):
            GAN_phase_1_domain_neg.train_on_batch(x_batch,y_batch_domain_inv)
            
 
        for _ in range(5):
            GAN_phase_1_domain_neg2.train_on_batch(x_batch,y_batch_domain_inv)
        
 
        if (i_step>1 and i_step % 100 == 0):
            
            prob_set = GAN_phase_1_task.predict(val_X)
            val_prob_set = prob_set
            print '---------------- validation --------------------------'
             
            dic_th_list = []
            i_prob = 0
            prob = prob_set[i_prob]
#             prob = np_output(prob)
            label_test = to_categorical(val_y_a[:, i_prob], CLASS_NUM[i_prob])             
            dic_th,_ = evaluate_result_valid(label_test, prob,i_step,'attr %d' % i_prob)
            dic_th_list.append(dic_th)
             
            
            
            prob_set = GAN_phase_1_task.predict(test_X)
            test_prob_set = prob_set
            print '---------------- test ----------------------'
 
            i_prob = 0
            prob = prob_set[i_prob]
#             prob = np_output(prob)
            label_test = to_categorical(test_y_a[:, i_prob], CLASS_NUM[i_prob])             
            evaluate_result_test(label_test, prob,i_step,'attr %d' % i_prob,dic_th_list[i_prob])
           
 
     
    return
         
     
if __name__ == "__main__":
 
    train()
    
    











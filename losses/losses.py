# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np
from keras import backend as K

def get_grad_norm(grad_r):
    grad_norm = 0.0
    for i_grad_r,e_grad_r in enumerate(grad_r):
        if e_grad_r is None:
            break
        else:
 
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


def gen_loss_sub(mask, log_prob, bias=0.):
    sample_weight = K.sum(mask, axis=1) + float(bias)
    sample_weight = K.maximum(0.0 * sample_weight, sample_weight)
    sample_weight = K.sign(sample_weight)

    mask_avg = mask / (K.sum(mask, axis=1, keepdims=True) + 1e-6)

    loss = - log_prob * mask_avg
    loss = K.sum(loss, axis=1)
    loss = K.sum(loss * sample_weight) / (K.sum(sample_weight) + 1e-6)

    return loss


def gen_loss_sub_unspv(log_prob, prob_target):
    loss = - log_prob * prob_target
    loss = K.sum(loss, axis=1)
    loss = K.mean(loss)

    return loss


def gen_meta_loss(xent_loss_pos_list, net, flag, use_loss_correction):
    grad_list = []
    grad_norm_list = []

    listOfVariableTensors = net.trainable_weights

    for r in range(len(xent_loss_pos_list)):
        grad_r = K.gradients(xent_loss_pos_list[r], listOfVariableTensors)
        grad_norm = get_grad_norm(grad_r)
        grad_list.append(grad_r)
        grad_norm_list.append(grad_norm)

    if use_loss_correction:
        r_set = [1, 2]
    else:
        r_set = [1]

    loss_grad_match = 0.
    for r in r_set:
        if flag:
            grad_list_0 = K.stop_gradient(grad_list[0])
            grad_norm_list_0 = K.stop_gradient(grad_norm_list[0])
        else:
            grad_list_0 = grad_list[0]
            grad_norm_list_0 = grad_norm_list[0]
        grad_prod = get_grad_prod(grad_list_0, grad_list[r])
        loss_grad_match += - grad_prod / (grad_norm_list_0 * grad_norm_list[r])

    return loss_grad_match

 

def loss_mi_sub(P1,P2,p1,p2,B):
    p1Tp2 = K.dot(K.transpose(p1),p2)
    P1TP2 = K.dot(K.transpose(P1),P2)/B
    mask = 1.-ada_eye(P1TP2)
    
    return P1TP2*mask*K.log((P1TP2*mask+1e-6)/(p1Tp2*mask+1e-6))

def loss_mi(P1,P2):
    
    B = ada_batch_size(P1)
    p1 = K.mean(P1,axis=1,keepdims=True)
    p2 = K.mean(P2,axis=1,keepdims=True)
    
    l1 = loss_mi_sub(P1,P2,p1,p2,B)
    l2 = loss_mi_sub(1.-P1,P2,1.-p1,p2,B)
    l3 = loss_mi_sub(P1,1.-P2,p1,1.-p2,B)
    l4 = loss_mi_sub(1.-P1,1.-P2,1.-p1,1.-p2,B)   
    
    return K.sum(l1+l2+l3+l4)

def MCC(P1,P2):
    
    B = ada_batch_size(P1)
    P1TP2 = K.dot(K.transpose(P1),P2)/B
    P1TP2 /= K.sum(P1TP2,axis=1,keepdims=True)
    P1TP2 /= my_get_shape(P1TP2)
    mask = 1.-ada_eye(P1TP2)
    
    return K.sum(K.abs(P1TP2*mask))
 
def orth_reg(W):
    WT = K.transpose(W)
    X = K.dot(WT,W)
    X = X *(1.- ada_eye(X))
    return 1e-8 * K.sum(X*X)

def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten

def sliced_wasserstein_distance_with_Theta(y_true, y_pred,Theta):
     
    y_true_proj = K.dot(y_true,K.transpose(Theta))
    y_pred_proj = K.dot(y_pred,K.transpose(Theta))
    y_true_proj = K.tf.contrib.framework.sort(y_true_proj,axis=0)
    y_pred_proj = K.tf.contrib.framework.sort(y_pred_proj,axis=0)
    loss = K.mean(K.abs(y_pred_proj-y_true_proj))
    
    return loss

def sliced_wasserstein_distance(y_true, y_pred):
    
    d = my_get_shape(y_pred)
    Theta = y_true[:,d:]
    y_true = y_true[:,:d]
    y_true_proj = K.dot(y_true,K.transpose(Theta))
    y_pred_proj = K.dot(y_pred,K.transpose(Theta))
    y_true_proj = K.tf.contrib.framework.sort(y_true_proj,axis=0)
    y_pred_proj = K.tf.contrib.framework.sort(y_pred_proj,axis=0)
    loss = K.mean(K.abs(y_pred_proj-y_true_proj))
    
    return loss
 
def my_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
 
def categorical_crossentropy(y_true, y_pred):
    return -K.mean(K.sum(y_true*K.log(y_pred+1e-6),axis=1), axis=-1)

def my_categorical_crossentropy(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true, y_pred))

def my_sparse_softmax_cross_entropy_with_logits(y_true, y_pred):
    loss = K.tf.nn.sparse_softmax_cross_entropy_with_logits(labels=K.tf.to_int32(y_true[:,0]),logits=y_pred)
    return K.mean(loss)

def my_sparse_softmax_cross_entropy_with_logits_inner_sub(y_true, y_pred):
    loss = K.tf.nn.sparse_softmax_cross_entropy_with_logits(labels=K.tf.to_int32(y_true[:,0]),logits=y_pred)
    return K.expand_dims(loss,axis=1)

def logit_sparce_cce_plus_pcce(y_true, y_pred):
    loss_cce = my_sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    loss_pcce = logit_pcce(y_true, y_pred)
    return loss_cce + loss_pcce

def prob_mean_squared_error(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l = -0.5 * K.mean(K.sum(K.square(y_pred - 0.5),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 0.5),axis=-1)
    return l 

def ori_prob_mean_squared_error(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l = - K.mean(K.sum(K.square(y_pred - 0.5),axis=1), axis=-1) +  K.sum(K.square(y_pred_mean - 0.5),axis=-1)
    return l 

def prob_mean_squared_error_g(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    c = my_get_shape(y_pred)
    weight = (K.random_binomial((1,),p=0.5) * 2. - 1.)
    l = 0.5 * K.mean( weight * K.sum(K.square(y_pred - 1./c),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 1./c),axis=-1)
    return l  

def logit_pmse_g(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    
    y_pred_mean = K.mean(y_pred,axis=0)
    c = my_get_shape(y_pred)
    weight = (K.random_binomial((1,),p=0.5) * 2. - 1.)
    l = 0.5 * K.mean( weight * K.sum(K.square(y_pred - 1./c),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 1./c),axis=-1)
    return l  

def max_entropy(y_true, y_pred):
    l = K.mean(K.sum(y_pred*K.log(y_pred+K.epsilon()),axis=1), axis=-1)
    return l

def logit_pcce(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    y_pred_mean = K.mean(y_pred,axis=0)
    l= -0.5 * K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) + 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l

def prob_categorical_crossentropy(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l= -0.5 * K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) + 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l 

def ori_prob_categorical_crossentropy(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l= - K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) +  K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l 

def prob_categorical_crossentropy_emph_kld(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    q = y_pred
    q_sum = K.sum(q,axis=0, keepdims=True)
    p = K.pow(q,2)/q_sum
    p = p/K.sum(p,axis=1, keepdims=True)
     
    l= - K.mean(K.sum(p*K.log(p+K.epsilon())-p*K.log(q+K.epsilon()),axis=1))  
    l+= K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l 

def neg_pmse(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l = 0.5 * K.mean(K.sum(K.square(y_pred - 0.5),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 0.5),axis=-1)
    return l 
def neg_pcce(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l= 0.5 * K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) + 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l

def logit_neg_pmse(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    
    y_pred_mean = K.mean(y_pred,axis=0)
    l = 0.5 * K.mean(K.sum(K.square(y_pred - 0.5),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 0.5),axis=-1)
    return l 
def logit_neg_pcce(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    
    y_pred_mean = K.mean(y_pred,axis=0)
    l= 0.5 * K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) + 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l

def pcce_tanh(y_true, y_pred):
    y_pred = (y_pred + 1.)/2.
    
    l = prob_categorical_crossentropy(y_true, K.clip(y_pred, K.epsilon(), 1) ) + prob_categorical_crossentropy(y_true, K.clip(1.-y_pred, K.epsilon(), 1))
    
    return l

def gaussian_activition_loss(y_true, y_pred):
      
    n = 512.
    y_pred_mean = K.mean(y_pred,axis=0)
    y_pred_sigma = K.dot(K.transpose(y_pred),y_pred)/n
    mask = 1. - ada_eye(y_pred_sigma)
    l = K.mean(K.square(y_pred_sigma*mask)) + K.mean(K.square(y_pred_mean))
    
    return l

 
def logit_sigmoid_cce(y_true, y_pred):
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) 

def logit_sigmoid_mse(y_true, y_pred):
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) 

def logit_sigmoid_cce_plus_pcce(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred_softmax)

def logit_sigmoid_mse_plus_pmse(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred_softmax)

def logit_sigmoid_cce_plus_neg_pcce(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred) + neg_pcce(y_true, y_pred_softmax)

def logit_sigmoid_mse_plus_neg_pmse(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred) + neg_pmse(y_true, y_pred_softmax)

def logit_cce(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.categorical_crossentropy(y_true, y_pred))

def inner_product_loss(y_true, y_pred):
    return K.mean(-K.sum(y_true*y_pred))

def logit_mse(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) 

def logit_cce_plus_pcce(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred)

def logit_mse_plus_pmse(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred)

def cce_plus_pcce(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true, y_pred)) + 0.01 * ori_prob_categorical_crossentropy(y_true, y_pred)

def cce_plus_mi(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true, y_pred)) +  loss_mi(y_pred, y_pred)

def mse_plus_pmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred)


def prob_binary_crossentropy(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l= -K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1)-K.mean(K.sum((1-y_pred)*K.log((1-y_pred)+1e-6),axis=1), axis=-1)  #+ K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l
 
def norm(x):
    return K.sqrt(K.sum(x*x))
def approx_entropy_pos(feats):
    batchSize = 32
    sign = 1
    s = 0.0
    cont = 0
    for i in range(batchSize):
        for j in range(i+1,batchSize):
            s = s + K.square(K.sum(feats[i]*feats[j])/(norm(feats[i])*norm(feats[j])+1e-6))
            cont = cont + 1
    s = sign*s/cont 
    return s

def approx_entropy_neg(feats):
    batchSize = 32
    
    s = 0.0
    cont = 0
    for i in range(batchSize):
        for j in range(i+1,batchSize):
            s = s +  K.sum(feats[i]*feats[j])/(norm(feats[i])*norm(feats[j])+1e-6)
            cont = cont + 1
    s = s/cont 
    return s
 

def my_kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(y_true * K.log(y_true / y_pred))

def my_balanced_kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return 0.5 * K.sum(y_true * K.log(y_true / y_pred), axis=-1) + 0.5 * K.sum(y_pred * K.log(y_pred / y_true), axis=-1)

def my_js_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    m = 0.5 * y_true + 0.5 * y_pred
    return 0.5 * K.sum(y_true * K.log(y_true / m), axis=1) + 0.5 * K.sum(y_pred * K.log(y_pred / m), axis=1)

def logit_focal_loss(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return focal_loss(y_true, y_pred) 

def focal_loss(y_true, y_pred):
    n_class = my_get_shape(y_pred) 
    gamma=2.
    alpha=1./n_class
 
    
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    loss = - (1. - y_pred)**gamma * y_true * K.log(y_pred) - alpha * y_pred**gamma * (1.-y_true) * K.log(1.-y_pred)  
    return K.sum(loss,axis=-1)

def focal_loss_single(y_true, y_pred):
     
    y_true = y_true[:,0]
    y_pred = y_pred[:,0]
     
    
    gamma=2.
    alpha=1./1000.

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    loss = - (1. - y_pred)**gamma * y_true * K.log(y_pred) - alpha * y_pred**gamma * (1.-y_true) * K.log(1.-y_pred)  
    return K.mean(loss)

def focal8pcce(y_true, y_pred):
    l1 = focal_loss_single(y_true, y_pred)
    l2 = prob_categorical_crossentropy(y_true, y_pred)
    return l1+l2

def normalize_vector(x):
     
    return K.l2_normalize(x)


def kld_(p, q):
     
    return my_kullback_leibler_divergence(p, q)

def tanh_kld_(p, q):
    
    p = (p+1.)/2.
    q = (q+1.)/2.
     
    return 0.5*my_kullback_leibler_divergence(p, q) + 0.5*my_kullback_leibler_divergence(q, p)

def logit_kld_(p, q):
    p = K.softmax(p,axis=1)
    q = K.softmax(q,axis=1)
     
    return my_kullback_leibler_divergence(p, q)

def kld2_(p, q):
     
    return mean_squared_error(p, q)

def l2normlized_dist(p,q):
    p = K.l2_normalize(p,axis=1)
    q = K.l2_normalize(q,axis=1)
    
    return mean_squared_error(p, q)


 

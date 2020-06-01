# -*- coding: utf-8 -*-

import torch
import numpy as np
import h5py
import os 
from sklearn.metrics import roc_auc_score
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
import time as time_simple
from util_keras import *
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.layers import *

import tensorflow as tf
from keras.engine import *

from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers
 
def label2uniqueID_sub(y):
    dic ={}
    uni_set = np.unique(y)
    y_new = y.copy()
    for i,e in enumerate(uni_set):
        y_new[y==e]=i
        dic[e] = i
 
    return y_new[:,np.newaxis],dic

def label2uniqueID(Y):
    dic_list = []
    for i in xrange(Y.shape[1]):
        y,dic = label2uniqueID_sub(Y[:,i])
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
        dic_list.append(dic)
 
    return new_Y,dic_list

def label2uniqueID_sub_test(y,dic):
   
    y_new = y.copy()
    for e in dic:
        y_new[y==e]=dic[e]

    return y_new[:,np.newaxis] 

def label2uniqueID_test(Y,dic_list):
    
    for i in range(Y.shape[1]):
        y = label2uniqueID_sub_test(Y[:,i],dic_list[i]) 
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
 
    return new_Y

def label2uniqueID_sub_train(y):
    dic ={}
    uni_set = np.unique(y)
    y_new = y.copy()
    for i,e in enumerate(uni_set):
        y_new[y==e]=i
        dic[e] = i
 
    return y_new[:,np.newaxis].astype(int),dic

def label2uniqueID_train(Y):
    dic_list = []
    for i in range(Y.shape[1]):
        y,dic = label2uniqueID_sub_train(Y[:,i])
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
        dic_list.append(dic)
 
    return new_Y,dic_list
 
def split_test_as_valid(test_y_c):
    
    unique_value = np.unique(test_y_c)
    idx = np.ones(test_y_c.shape[0])
    for i_uni,e_uni in enumerate(unique_value):
        a = idx[np.where(test_y_c[:,0]==e_uni)]
        idx[np.where(test_y_c[:,0]==e_uni)] = a*np.round(np.random.rand(len(a)))
        
    idx_pos = idx
    idx_neg = 1-idx
    
    return idx_pos.astype(bool),idx_neg.astype(bool)

 

def split_train_test(train_y_a,test_rate = 0.2):
    
    user_id_set = np.unique(train_y_a[:,0])
    user_age_set = np.array([train_y_a[train_y_a[:,0]==e,1][0] for e in user_id_set])
    test_user_id_set = []
    for age in np.unique(user_age_set):
      
        user_id_subset = user_id_set[user_age_set == age].copy()
        np.random.shuffle(user_id_subset)
        test_user_id_set.extend(user_id_subset[:int(np.ceil(len(user_id_subset)*test_rate))])
    print('user_id_set,test_user_id_set',user_id_set,test_user_id_set)
    train_user_id_set = np.setdiff1d(user_id_set,test_user_id_set)
 
    return train_user_id_set,test_user_id_set
 
def argmin_mean_FRR_st_FAR(label_test, prob,exp_far=0.01,pen_far=10.):
    
    def f(e):
        label_out = prob > e
        far = FAR_score(label_test, label_out)
        frr = FRR_score(label_test, label_out)
        return frr + pen_far * np.maximum(0,far-exp_far)
    res = minimize_scalar(f, bounds=(prob.min(), prob.max()), method='bounded')
     
    return res.x

def argmin_fixFRR(label_test, prob,exp_frr=0.05):
    
    def f(e):
        label_out = prob > e
        far = FAR_score(label_test, label_out)
        frr = FRR_score(label_test, label_out)
        return np.abs(frr - exp_frr)
    if exp_frr < 0.05:
        res = minimize_scalar(f, bounds=(-1., 2.), method='bounded')
    else:
        res = minimize_scalar(f, bounds=(prob.min(), prob.max()), method='bounded')
    
     
    return res.x

def argmin_fixFAR(label_test, prob,exp_far=0.05):
    
    def f(e):
        label_out = prob > e
        far = FAR_score(label_test, label_out)
        return np.abs(far - exp_far)
    res = minimize_scalar(f, bounds=(prob.min(), prob.max()), method='bounded')
     
    return res.x

def argmin_mean_FAR_FRR(label_test, prob):
    
    def f(e):
        label_out = prob > e
        far = FAR_score(label_test, label_out)
        frr = FRR_score(label_test, label_out)
        return np.abs(far-frr)
    res = minimize_scalar(f, bounds=(prob.min(), prob.max()), method='bounded')
     
    return res.x

def FAR_score_torch(y_true,y_pred):
    
    div = torch.sum(y_pred[y_true==0]).float()/(torch.sum(y_true==0).float()+1e-6)
    
    return div.cpu().numpy()
 
def FRR_score_torch(y_true,y_pred):
    div = torch.sum(y_pred[y_true==1]==0).float()/(torch.sum(y_true==1).float()+1e-6)
    return div.cpu().numpy()

def argmin_fixFAR_torch(label_test, prob,exp_far=0.05):
    prob_min = torch.min(prob).cpu().numpy()
    prob_max = torch.max(prob).cpu().numpy()
    def f(e):
        label_out = prob > e
        far = FAR_score_torch(label_test, label_out)
        return np.abs(far - exp_far)
    res = minimize_scalar(f, bounds=(prob_min, prob_max), method='bounded')
     
    return res.x

def argmin_mean_FAR_FRR_torch(label_test, prob):
    prob_min = torch.min(prob).cpu().numpy()
    prob_max = torch.max(prob).cpu().numpy()
    def f(e):
        label_out = prob > e
        far = FAR_score_torch(label_test, label_out)
        frr = FRR_score_torch(label_test, label_out)
        return np.abs(far-frr)
    res = minimize_scalar(f, bounds=(prob_min, prob_max), method='bounded')
     
    return res.x

  
def auc_MTL(label_test, prob):
    if np.ndim(prob) == 1:
        auc_all = roc_auc_score(label_test, prob)

        return auc_all, auc_all
    else:
        valid_task_id = []
        for i in range(label_test.shape[1]):
            if len(np.unique(label_test[: ,i])) > 1:
                valid_task_id.append(i)
        task_num = len(valid_task_id)
        auc_all = []
        for i_order in valid_task_id:
            auc_all.append(roc_auc_score(label_test[:, i_order], prob[:, i_order]))
        auc_all = np.array(auc_all)

        return np.mean(auc_all), auc_all
     
def FAR_score(y_true,y_pred):
    
    return np.sum(y_pred[np.where(y_true==0)])/float(np.sum(y_true==0))
 
def FRR_score(y_true,y_pred):
    
    return np.sum(y_pred[np.where(y_true==1)]==0)/float(np.sum(y_true==1))
 
def evaluate_result_valid_simple(label_test, prob,i,str_input):
    label_out = np.round(prob)
    mean_auc, auc_all = auc_MTL(label_test, prob)
 
    acc = np.mean(np.argmax(label_test,axis=1) == np.argmax(prob,axis=1))
    
    Y_test = label_test
    train_labels = np.unique(np.argmax(Y_test,axis=1))
    n_user_train = len(train_labels)


    print('%s: step %d, auc %g, acc %g, n_eval %d' % (str_input, i,  np.mean(auc_all),acc,n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
    print(auc_all)
    
    
    return auc_all
     
def evaluate_result_valid(label_test, prob,i,str_input):
    
    mean_auc, auc_all = auc_MTL(label_test, prob)
 
    Y_test = label_test
    train_labels = np.unique(np.argmax(Y_test,axis=1))
    n_user_train = len(train_labels)

    dic_th = {}
    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    for i_user,idx in enumerate(train_labels):
        dic_th[idx] = argmin_mean_FAR_FRR(Y_test[:,idx], prob[:,idx])
        label_out  = prob[:,idx] > dic_th[idx]
        FAR_ALL[i_user] = FAR_score(Y_test[:,idx],label_out )
        FRR_ALL[i_user] = FRR_score(Y_test[:,idx],label_out )
        ACC_ALL[i_user] = np.mean(Y_test[:,idx] == label_out )
        
    acc = np.mean(np.argmax(Y_test,axis=1) == np.argmax(prob,axis=1))


    print('%s: step %d, auc %g, FAR %g, FRR %g, acc %g, acc_avg %g, n_eval %d' % (str_input, i,  np.mean(auc_all), np.mean(FAR_ALL),np.mean(FRR_ALL),acc,np.mean(ACC_ALL),n_user_train))
 
    return dic_th,np.mean(auc_all)

def evaluate_result_test(label_test, prob,i,str_input,dic_th):
 
    mean_auc, auc_all = auc_MTL(label_test, prob)

    Y_test = label_test
    train_labels = np.unique(np.argmax(Y_test,axis=1))
    n_user_train = len(train_labels)

    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    for i_user,idx in enumerate(train_labels):
        label_out  = prob[:,idx] > dic_th.get(idx,0.5)
        FAR_ALL[i_user] = FAR_score(Y_test[:,idx],label_out )
        FRR_ALL[i_user] = FRR_score(Y_test[:,idx],label_out )
        ACC_ALL[i_user] = np.mean(Y_test[:,idx] == label_out )
        
    acc = np.mean(np.argmax(Y_test,axis=1) == np.argmax(prob,axis=1))


    print('%s: step %d, auc %g, FAR %g, FRR %g, acc %g, acc_avg %g, n_eval %d' % (str_input, i,  np.mean(auc_all), np.mean(FAR_ALL),np.mean(FRR_ALL),acc,np.mean(ACC_ALL),n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
#     print auc_all

    return
  

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process

import multiprocessing
from itertools import product
from contextlib import contextmanager

def sub_fun(i_user,Y_test,prob):
    idx = i_user
    dic_th = argmin_mean_FRR_st_FAR(Y_test, prob)

    label_out = prob > dic_th
    FAR_ALL = FAR_score(Y_test,label_out)
    FRR_ALL = FRR_score(Y_test,label_out)
    ACC_ALL = np.mean(Y_test == label_out)
    return FAR_ALL,FRR_ALL,ACC_ALL,dic_th

def sub_fun_unpack(args):
    return sub_fun(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

 
 

def sub_fun_test(i_user,Y_test,prob,dic_th):
    idx = i_user
 
    label_out = prob > dic_th 
    FAR_ALL = FAR_score(Y_test,label_out)
    return FAR_ALL 

def sub_fun_test_unpack(args):
    return sub_fun_test(*args)
 

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def my_zscore(X):
    mean_X = np.mean(X, 0)
    std_X = np.std(X, 0)
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X, mean_X, std_X


def my_zscore_test(X, mean_X, std_X):
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X
 
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    

def read_h5(file_path):
    f = h5py.File(file_path,'r')
    x = np.array(f['X'])
    f.close()
    return x
def read_csv(file_path):
    df = pd.read_csv(file_path,header=None, encoding='utf-8-sig')   
    x = df.values
    return x

def idx_in_subset(subset,y):
    fullset = np.unique(y)
    # diffset = np.setdiff1d(fullset,subset)
    dic_binary = {e:0 for e in fullset}
    for e in subset:
        dic_binary[e] = 1
    idx = np.array(map(lambda x: dic_binary[x],y))
    return idx == 1
 
def int2bin(a,bin_size):
    return [int(x) for x in bin(int(a))[2:].zfill(bin_size)]

 

from sklearn import manifold

from scipy.special import logsumexp

def softmax(x,axis=1):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
 

def to_categorical(y,nb_classes):
    return np.eye(nb_classes)[y]

  

def get_prior(y):
    uni_set = np.unique(y)
    n = len(uni_set)
    p = np.zeros(n)
    for i,e in enumerate(uni_set):
        p[i] = np.sum(y==e)
    return p/np.sum(p)
 

from time import sleep
import sys
def compute_fun():
    a = np.random.random((3,3))
    for _ in range(100):
        a = np.dot(a,a)
        sys.stdout.write('.')
    sleep(1.)
    print('.')

    return

def get_md5(src):
    import hashlib
    m2 = hashlib.md5()
    m2.update(src)
    return m2.hexdigest()


def get_timestamp_ms():
    import time
    return int(time.time() * 1000)





def get_mac_address():
    import uuid
    node = uuid.getnode()
    return uuid.UUID(int=node).hex[-12:]


def get_today():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d")

 

def split_uniid(uniid):
    
    unique_uniid = np.unique(uniid)
    idx = np.arange(len(unique_uniid))
    np.random.shuffle(idx)
    idx_select = unique_uniid[idx[:len(unique_uniid)/2]]
    idx = idx_in_subset(idx_select,uniid)
    
    return idx == 1
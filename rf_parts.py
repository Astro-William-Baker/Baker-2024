#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:29:26 2021

@author: will
"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
import numpy as np
from hyp_param_tuning import hyp_test
from sklearn.inspection import permutation_importance
from astropy.stats import bootstrap
import random
import pandas as pd

def boots_RFR(data, n_est=300, min_samp_leaf=6, max_dep=30, n_cores=4, t_size=0.2, n_times=100):
    res=[]
    print('started bootstrap')
    for i in range(n_times):
        new_data=resample(data.T, replace=True).T
        target=new_data[0]
        features=new_data[1:]

        features=features.T
        scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0),
                                        with_scaling=True,
                                        copy=True)  
        
        scaled_features = scaler.fit_transform(features)
        features_tr, features_va, target_tr, target_va = train_test_split(scaled_features,
                                                                      target,
                                                                      test_size=0.2)    
        reg = RandomForestRegressor(n_estimators=n_est,       
                                 min_samples_leaf=min_samp_leaf ,max_depth=max_dep ,  
                                 n_jobs=n_cores) 
        reg.fit(features_tr, target_tr) 
        performance = reg.feature_importances_
        res.append(performance)
        if (i % 20 )== 0:
            print('Bootstrap number : {}'.format(i))
  
       
    res=-np.array(res)
    
    error=np.var(res, axis=0)
    print('errors obtained')
    return error


def boots_RFR_updated(data, n_est=300, min_samp_leaf=6, max_dep=30, n_cores=4, t_size=0.2, n_times=100, wide=True):
    res=[]
    print('started bootstrap')
    for i in range(n_times):
        new_data=data
        #leng=len(data[0])
        #df=pd.DataFrame(new_data)
        #new_data=df.sample(n=leng, replace=True)
        #new_data=bootstrap(data.T, bootnum=1).T
        new_data=resample(data.T, replace=True).T
        #print(new_data)
        target=new_data[0]
        features=new_data[1:]

        features=features.T
        scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0),
                                        with_scaling=True,
                                        copy=True)  
        
        scaled_features = scaler.fit_transform(features)
        features_tr, features_va, target_tr, target_va = train_test_split(scaled_features,
                                                                      target,
                                                                      test_size=t_size)    
        reg = RandomForestRegressor(n_estimators=n_est,       
                                 min_samples_leaf=min_samp_leaf ,max_depth=max_dep ,  
                                 n_jobs=n_cores) 
        reg.fit(features_tr, target_tr) 
        performance = reg.feature_importances_
        res.append(performance)
        if (i % 20 )== 0:
            print('Bootstrap number : {}'.format(i))

    #actual performance
    target=data[0]
    features=data[1:]
    features=features.T
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0),
                                    with_scaling=True,
                                    copy=True)  
    scaled_features = scaler.fit_transform(features)
    features_tr, features_va, target_tr, target_va = train_test_split(scaled_features,
                                                                  target,
                                                                  test_size=t_size)    
    reg = RandomForestRegressor(n_estimators=n_est,       
                             min_samples_leaf=min_samp_leaf ,max_depth=max_dep ,  
                             n_jobs=n_cores) 
    reg.fit(features_tr, target_tr) 
    actual_performance = reg.feature_importances_
    #end of actual performance
  
       
    res=np.array(res)
    #real=actual_performance
    real=np.median(res, axis=0)
    print(real)
    #error=np.var(res, axis=0)
    if wide==True:
        upper=np.quantile(res,0.95,axis=0)
        lower=np.quantile(res,0.05,axis=0)
    else:
        upper=np.quantile(res,0.84,axis=0)
        lower=np.quantile(res,0.16,axis=0)
    error_upp=np.abs(np.subtract(upper,real))
    error_low=np.abs(np.subtract(real,lower))
    errors=np.array([error_low, error_upp])
    print(errors)
    print('errors obtained')
    return real, errors

def true_boots_RFR(data, n_est=300, min_samp_leaf=6, max_dep=30, n_cores=4, t_size=0.2, n_times=100):
    res=[]
    print('started bootstrap')
    for i in range(n_times):
        new_data=resample(data, replace=True)
        target=new_data[0]
        features=new_data[1:]

        features=features.T
        scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0),
                                        with_scaling=True,
                                        copy=True)  
        
        scaled_features = scaler.fit_transform(features)
        features_tr, features_va, target_tr, target_va = train_test_split(scaled_features,
                                                                      target,
                                                                      test_size=0.2)    
        reg = RandomForestRegressor(n_estimators=n_est,       
                                 min_samples_leaf=min_samp_leaf ,max_depth=max_dep ,  
                                 n_jobs=n_cores) 
        reg.fit(features_tr, target_tr) 
        performance = reg.feature_importances_
        res.append(performance)
        if (i % 20 )== 0:
            print('Bootstrap number : {}'.format(i))
  
       
    res=-np.array(res)
    upper=np.percentile(res, 84, axis=0)
    lower=np.percentile(res, 16, axis=0)
    #error=np.var(res, axis=0)
    error=np.array((lower, upper))
    print('errors obtained')
    return error



def basic_RF(data, n_est=300, min_samp_leaf=6, max_dep=30, n_cores=4, t_size=0.2 ,param_check=False):
   
    
    #quantity to be predicted
    target=data[0]
    #quantities to use in predicting
    features=data[1:]

    features=features.T

    #resize data set
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0),
                                        with_scaling=True,
                                        copy=True)   
                                                  
    scaled_features = scaler.fit_transform(features)
    #split dataset set into training data and target data
    features_tr, features_va, target_tr, target_va = train_test_split(scaled_features,
                                                                      target,
                                                                      test_size=t_size)
        
    reg = RandomForestRegressor(n_estimators=n_est,       
                                 min_samples_leaf=min_samp_leaf, max_depth=max_dep ,  
                                 n_jobs=n_cores) 
    if param_check==True:
        hyp_test(reg, features_va, target_va)
        
    
    reg.fit(features_tr, target_tr)  
    
    pred_tr = reg.predict(features_tr)    # an array of predictions for each object in the 
                                          # training sample 
    
    pred_va = reg.predict(features_va)   # an array of predictions for each object in the 
                                         # validation sample 
    
    mse_tr=np.mean(np.power(pred_tr - target_tr, 2))
    mse_va=np.mean(np.power(pred_va - target_va, 2))
                                       
                                        
    performance = reg.feature_importances_
    
    return performance, mse_tr, mse_va, features_tr

def diff_error_test_RF(data, z, z_error, dev=0.5):
    z=z
    z_vals=[]
    for i in range(len(z)):
        z_vals.append(z[i]+np.random.normal(0,dev*z_error[i]))

def permutate(data, n_est=300, min_samp_leaf=6, max_dep=30, n_cores=4, t_size=0.2 ,param_check=False):

      #quantity to be predicted
    target=data[0]
    #quantities to use in predicting
    features=data[1:]

    features=features.T

    #resize data set
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0),
                                        with_scaling=True,
                                        copy=True)   
                                                  
    scaled_features = scaler.fit_transform(features)
    #split dataset set into training data and target data
    features_tr, features_va, target_tr, target_va = train_test_split(scaled_features,
                                                                      target,
                                                                      test_size=t_size)
        
    reg = RandomForestRegressor(n_estimators=n_est,       
                                 min_samples_leaf=min_samp_leaf, max_depth=max_dep ,  
                                 n_jobs=n_cores) 
    if param_check==True:
        hyp_test(reg, features_va, target_va)
        
    
    reg.fit(features_tr, target_tr)  
    
    pred_tr = reg.predict(features_tr)    # an array of predictions for each object in the 
                                          # training sample 
    
    pred_va = reg.predict(features_va)   # an array of predictions for each object in the 
                                         # validation sample 
    
    mse_tr=np.mean(np.power(pred_tr - target_tr, 2))
    mse_va=np.mean(np.power(pred_va - target_va, 2))
                                       
                                        
    performance = reg.feature_importances_


    result_tr = permutation_importance(reg, features_tr, target_tr, n_repeats=30,
                                    random_state=42, n_jobs=n_cores)
    sorted_idx_tr = result_tr.importances_mean.argsort()[::-1]

    
    result_va = permutation_importance(reg, features_va, target_va, n_repeats=30,
                                random_state=42, n_jobs=n_cores)
    #sorted_idx_va = result_va.importances_mean.argsort()[::-1]
  

    return performance, mse_tr, mse_va, features_tr, result_tr.importances[sorted_idx_tr].T, result_va.importances[sorted_idx_tr].T

class permutation_(object):

    def __init__(self, data, n_est=300, min_samp_leaf=6, max_dep=30, n_cores=4, t_size=0.2 ,param_check=False):
        super(permutation_,self).__init__()

        target=data[0]
        #quantities to use in predicting
        features=data[1:]

        features=features.T

        #resize data set
        scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0),
                                            with_scaling=True,
                                            copy=True)   
                                                      
        scaled_features = scaler.fit_transform(features)
        #split dataset set into training data and target data
        features_tr, features_va, target_tr, target_va = train_test_split(scaled_features,
                                                                          target,
                                                                          test_size=t_size)
            
        reg = RandomForestRegressor(n_estimators=n_est,       
                                     min_samples_leaf=min_samp_leaf, max_depth=max_dep ,  
                                     n_jobs=n_cores) 
        if param_check==True:
            hyp_test(reg, features_va, target_va)
            
        
        reg.fit(features_tr, target_tr)  
        
        pred_tr = reg.predict(features_tr)    # an array of predictions for each object in the 
                                              # training sample 
        
        pred_va = reg.predict(features_va)   # an array of predictions for each object in the 
                                             # validation sample 
        
        mse_tr=np.mean(np.power(pred_tr - target_tr, 2))
        mse_va=np.mean(np.power(pred_va - target_va, 2))
                                           
                                            
        performance = reg.feature_importances_


        result_tr = permutation_importance(reg, features_tr, target_tr, n_repeats=30,
                                        random_state=42, n_jobs=n_cores)
        sorted_idx_tr = result_tr.importances_mean.argsort()[::-1]

        
        result_va = permutation_importance(reg, features_va, target_va, n_repeats=30,
                                    random_state=42, n_jobs=n_cores)
        #sorted_idx_va = result_va.importances_mean.argsort()[::-1]
        self.performance=performance
        self.mse_tr= mse_tr
        self.mse_va= mse_va 
        self.features_tr= features_tr, 
        self.result_tr= result_tr.importances_mean
        self.result_va= result_va.importances_mean
        self.error_tr= result_tr.importances_std
        self.error_va= result_tr.importances_std

        self.idx=sorted_idx_tr


  


if __name__ == "__main__": 
        rf=1
        
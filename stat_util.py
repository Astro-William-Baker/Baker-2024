#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:09:36 2021

@author: will

selection of tools that are useful including code to remove NaNs and outliers
"""

__author__='William Baker'





import numpy as np
import matplotlib.pyplot as plt
#import statsmodels.api as sm



def any_nans(y):
    b = True if True in np.isnan(np.array(y)) else False
    if b==True:
        print(b)
    else:
        print(b)
        
class con_nans():
    def __init__(self,y):
        self.y=y
    def __call__(self):
        b = True if True in np.isnan(np.array(self.y)) else False
        if b==True:
            print(b)
        else:
            print(b) 

def outlier_cut_old(data, dev=3):
    data=np.array(data)
    print(data.shape)
    #data=data.reshape((data.shape[0],data.shape[2]))
    dold=data
    for i in range(len(data)):
        
            av=np.mean(data[i])
            sd=np.std(data[i])
    
            lower=av-dev*sd
            upper=av+dev*sd
      
            index=np.where((data[i]<upper) & (data[i]>lower))
         
            data1=data[:,index]
            data=data1
            data=data.reshape((data.shape[0],data.shape[2]))
        
    print('{} outliers removed'.format(dold.shape[1]-data.shape[1]))   
    return data
    
            
        

def outlier_cut(data, dev=3):
    data=np.array(data)
    print(data.shape)
    #data=data.reshape((data.shape[0],data.shape[2]))
    dold=data
    for i in range(len(data)):
        
            av=np.median(data[i])
            sd=np.std(data[i])
    
            lower=av-dev*sd
            upper=av+dev*sd
      
            index=np.where((data[i]<upper) & (data[i]>lower))
         
            data1=data[:,index]
            data=data1
            data=data.reshape((data.shape[0],data.shape[2]))
        
    print('{} outliers removed'.format(dold.shape[1]-data.shape[1]))   
    return data

class outliers(object):
    def __init__(self, data):
        self.data=data
    
    def oc(self, dev=3):
        data=np.array(self.data)
        print(data.shape)
        #data=data.reshape((data.shape[0],data.shape[2]))
        dold=data
        for i in range(len(data)):
            
                av=np.mean(data[i])
                sd=np.std(data[i])
        
                lower=av-dev*sd
                upper=av+dev*sd
          
                index=np.where((data[i]<upper) & (data[i]>lower))
             
                data1=data[:,index]
                data=data1
                data=data.reshape((data.shape[0],data.shape[2]))
            
        print('{} outliers removed'.format(dold.shape[1]-data.shape[1])) 
        return data

    def oc_keep(self, keeper, dev=3):
        try:
            data=np.array(self.data)
            print(data.shape)
            #data=data.reshape((data.shape[0],data.shape[2]))
            dold=data
            for i in range(len(data)):
                    #average
                    av=np.mean(data[i])
                    #stadard deviation
                    sd=np.std(data[i])
            
                    lower=av-dev*sd
                    upper=av+dev*sd
              
                    index=np.where((data[i]<upper) & (data[i]>lower))
                 
                    data1=data[:,index]
                    keeper=keeper[index]
                    data=data1
                    data=data.reshape((data.shape[0],data.shape[2]))
                
            print('{} outliers removed'.format(dold.shape[1]-data.shape[1])) 
        except:
            print('Unable to remove outliers')
        return data, keeper

def rem_nans(data):
    data=np.array(data)
    dold=data
    for i in range(len(data)):
        
            index=np.where(~np.isnan(data[i]) == True)
         
            data1=data[:,index]
            data=data1
            data=data.reshape((data.shape[0],data.shape[2]))
        
    print('{} NaNs removed'.format(dold.shape[1]-data.shape[1])) 
    return data

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return 'err'


class qq_test():

    def __init__(self, data):
        fig, ax = plt.subplots(nrows=1, ncols=data.shape[0])
        for i in range(data.shape[0]):
            sm.qqplot(data[i], line='45', fit=True, ax=ax[i])

def func_timer(func):
    """Times how long the function took."""

    def f(*args, **kwargs):
        import time
        start = time.time()
        results = func(*args, **kwargs)
        print("Elapsed: %.2fs" % (time.time() - start))
        return results

    return f


        


if __name__ == "__main__": 

    '''x=[0,1,4,5,7,np.nan, 8 , np.nan]
    y=[0,1,4,5,7,np.nan, 8 , 3]
    z=[0,np.nan,4,5,7,np.nan, 8 , np.nan]
    y1=[np.nan,1,4,5,7,np.nan, 8 , 3]
    z1=[0,np.nan,4,5,7,np.nan, 8 , np.nan]
    
    dats=[x,y,z,y1,z1]
    
    o=nan_destroy(dats)
    
    print(o)'''
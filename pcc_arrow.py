#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:05:22 2021

@author: will
"""


__author__='William M. Baker'


import numpy as np
import pingouin as pg
import pandas as pd
from sklearn.utils import resample
from scipy import odr



class pcc(object):
    '''calculates partial correlation coefficients between 3 quantities'''
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
        data={'y': self.y, 'x': self.x, 'z':self.z}
        df=pd.DataFrame(data,columns=['y','x','z'])
        res=pg.partial_corr(data=df, x='x', y='y', covar=[ 'z'], method='spearman').round(3)
        res1=pg.partial_corr(data=df, x='y', y='z', covar=['x'], method='spearman').round(3)
        res2=pg.partial_corr(data=df, x='z', y='x', covar=['y'], method='spearman').round(3)
        self.r=pd.to_numeric(res['r'][0])
        self.r1=pd.to_numeric(res1['r'][0])
        self.r2=pd.to_numeric(res2['r'][0])
    def pcc_res(self):
        print(self.r)
        print(self.r1)
        print(self.r2)
        return None
    
class arro(pcc):
    '''calculates the angle theta of the partial correlation coefficient vector then calculates dx and dy 
    associated with it - x1 and x2 are the centre of the arrow and are chosen to be the median value of the 
    quantity involved''' 
    def arr_setup(self):
        if self.r1<=0:
            theta=np.arctan(self.r2/self.r1)+np.pi
        else:
            theta=np.arctan(self.r2/self.r1)
        self.angle=(theta/np.pi *180).round(1)
        print(self.angle)
        width = lambda a: max(a)-min(a)
        if width(self.x) > 2.:
            rad=1
        else:
            rad=1 
        
        dx=rad*np.sin(theta)
        dy=rad*np.cos(theta)
        ratio=width(self.x)/width(self.y)
        dy= dy/ratio
        x1=np.median(self.x)
        y1=np.median(self.y)
        return dx, dy, x1, y1
    
class arrow_error(object):
    def __init__(self, x ,y ,z, n_times=100):
        self.x=x
        self.y=y
        self.z=z
        data={'y': self.y, 'x': self.x, 'z':self.z}
        df=pd.DataFrame(data,columns=['y','x','z'])
        res=pg.partial_corr(data=df, x='x', y='y', covar=[ 'z'], 
            method='spearman').round(3)
        yz_x=pg.partial_corr(data=df, x='y', y='z', covar=['x'], 
            method='spearman').round(3)
        zx_y=pg.partial_corr(data=df, x='z', y='x', covar=['y'], 
            method='spearman').round(3)
        self.r=pd.to_numeric(res['r'][0])
        self.r1=pd.to_numeric(yz_x['r'][0])
        self.r2=pd.to_numeric(zx_y['r'][0])
        if self.r1<=0:
            theta=np.arctan(self.r2/self.r1)+np.pi
        else:
            theta=np.arctan(self.r2/self.r1)
        self.angle=(theta/np.pi *180).round(3)
        sig=[]
        for i in range(n_times):
            u,v,w,=resample(self.x, self.y, self.z, replace=True, 
                n_samples=len(self.x))
            u,v,w=np.array(u), np.array(v), np.array(w)
            data1={'y': v, 'x': u, 'z':w}
            df1=pd.DataFrame(data1,columns=['y','x','z'])
            rho1=pd.to_numeric(pg.partial_corr(data=df1, x='y', y='z',
             covar=['x'], method='spearman').round(3)['r'][0])
            rho2=pd.to_numeric(pg.partial_corr(data=df1, x='z', y='x', 
             covar=['y'], method='spearman').round(3)['r'][0])
            if rho1<=0:
                theta1=np.arctan(rho2/rho1)+np.pi
            else:
                theta1=np.arctan(rho2/rho1)
            sig.append((theta1/np.pi *180))
        #self.angle_error=np.absolute(np.mean(sig)-self.angle)
        self.angle_error=np.std(sig)
        self.angle_rad=-self.angle+90
    
    def __str__(self):
        return "Arrow Angle = {}, Error = {}".format(self.angle.round(2),
         self.angle_error.round(2))
        
        

class arrow_error_new(object):

    def __init__(self, x ,y ,z, n_times=100, old=False):
        
        self.x=x
        self.y=y
        self.z=z
        data={'y': self.y, 'x': self.x, 'z':self.z}
        df=pd.DataFrame(data,columns=['y','x','z'])
        res=pg.partial_corr(data=df, x='x', y='y', covar=[ 'z'], 
            method='spearman').round(3)
        yz_x=pg.partial_corr(data=df, x='y', y='z', covar=['x'], 
            method='spearman').round(3)
        zx_y=pg.partial_corr(data=df, x='z', y='x', covar=['y'], 
            method='spearman').round(3)
        self.r=pd.to_numeric(res['r'][0])
        self.r1=pd.to_numeric(yz_x['r'][0])
        self.r2=pd.to_numeric(zx_y['r'][0])
        if self.r1<=0:
            #theta=np.arctan(self.r2/self.r1)+np.pi
            theta=np.pi/2-np.arctan(self.r1/self.r2)  #+np.pi
        elif self.r2<=0:
            theta=np.pi/2-np.arctan(self.r1/self.r2) + np.pi  #+np.pi
        elif self.r1<=0 and self.r2<=0:
            theta=np.arctan(self.r1/self.r2)

        else:
            #theta=np.arctan(self.r2/self.r1)
            theta=np.pi/2-np.arctan(self.r1/self.r2)

        if old==True:
            theta=np.arctan(self.r2/self.r1) #+np.pi
        #print(self.r1)
        #print(self.r2)
        #print(theta)
        self.theta=theta
        self.angle=(theta/np.pi *180).round(3)
        sig=[]
        for i in range(n_times):
            u,v,w,=resample(self.x, self.y, self.z, replace=True, 
                n_samples=len(self.x))
            u,v,w=np.array(u), np.array(v), np.array(w)
            data1={'y': v, 'x': u, 'z':w}
            df1=pd.DataFrame(data1,columns=['y','x','z'])
            rho1=pd.to_numeric(pg.partial_corr(data=df1, x='y', y='z',
             covar=['x'], method='spearman').round(3)['r'][0])
            rho2=pd.to_numeric(pg.partial_corr(data=df1, x='z', y='x', 
             covar=['y'], method='spearman').round(3)['r'][0])
            if rho1<=0:
                theta1=np.arctan(rho2/rho1)+np.pi
            else:
                theta1=np.arctan(rho2/rho1)
            sig.append((theta1/np.pi *180))
        #self.angle_error=np.absolute(np.mean(sig)-self.angle)
        self.angle_error=np.std(sig)
        self.angle_rad=-self.angle+90
    
    def __str__(self):
        return "Arrow Angle = {}, Error = {}".format(self.angle.round(2),
         self.angle_error.round(2))

    def dim(self):
        width = lambda a: max(a)-min(a)
        theta=self.theta
        if width(self.x) > 2.:
            rad=1
        else:
            rad=1 
        
        dx=rad*np.sin(theta)
        dy=rad*np.cos(theta)
        ratio=width(self.x)/width(self.y)
        dy= dy/ratio
        x1=np.median(self.x)
        y1=np.median(self.y)
        return dx, dy, x1, y1

def odrfunc(x,y):
    '''Orthogonal distance regression function with residual standard deviation'''
    linear_func = lambda B, x: B[0]*x +B[1]
    linear=odr.Model(linear_func)
    mydata = odr.RealData(x, y)
    myodr = odr.ODR(mydata, linear, beta0=[1., 2.])

    myoutput = myodr.run()
    #myoutput.pprint()
    r=myoutput.beta
    rerr=myoutput.sd_beta
    re=myoutput.res_var
    
    std=np.sqrt(re)
    return r, rerr, std

if __name__ == "__main__": 
    ab=1
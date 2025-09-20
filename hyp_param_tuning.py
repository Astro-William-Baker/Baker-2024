#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:11:01 2021

@author: will
"""

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import time

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            

def hyp_test(model, X, y, grid=True,def_params=False, n_cores=4):
        start = time.time()
        if def_params==False:
            param_grid={'max_depth': [10,30,50,70,100,200],
                  'n_estimators': [50,100,200,300,400,600,800],
                  'min_samples_leaf': [3, 6,15,30,60,120,180]}
        else:
            pass
        
        if grid == True:
            grid_search=GridSearchCV(model, param_grid=param_grid, cv=4, n_jobs=n_cores)
            grid_search.fit(X,y)
            print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
                  % (time.time() - start, len(grid_search.cv_results_['params'])))
            report(grid_search.cv_results_)
            #print('Best Esimator: %.2f' %grid_search.best_estimator_)
        else:
            n_iter_search=30
            sh = RandomizedSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_iter=n_iter_search, n_jobs=n_cores)
            sh.fit(X, y)
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time.time() - start), n_iter_search))
            report(sh.cv_results_)
       
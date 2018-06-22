# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:00:00 2018

@author: fabrice.lallement
"""

class LocalizationEstimator():
    
    def __init__(self, dt):
        self.dt = dt
    
    def compute_localization(self):
        raise NotImplementedError("Must be implemented in sub-classes")
        

        
if __name__ == '__main__':
    
    estimator = LocalizationEstimator()
    
    estimator.computeLocalization()
    
    
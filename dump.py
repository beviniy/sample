# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 19:44:02 2016

@author: Administrator
"""

"""
Simple demo of a scatter plot.
"""
import numpy as np
import cPickle as pickle


if __name__ == '__main__':
    path = 'features/'
    name = 'feature_all.pkl'
    
    f = open(path+name)
    features = pickle.load(f)
    f.close()
    
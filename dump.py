# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 19:44:02 2016

@author: Administrator
"""

import cPickle as pickle

f = open('dump.pkl')
data = pickle.load(f)
f.close()
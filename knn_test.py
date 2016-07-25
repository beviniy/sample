# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:52:47 2016

@author: Administrator
"""

'''
from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
'''


from sklearn.neighbors import KDTree
import numpy as np


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=50, metric='euclidean')
kdt.query(X, k=2, return_distance=False)
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:50:04 2016

@author: Administrator
"""

import os
from room_type import ROOM_TYPE_CHOICES
import glob
import pickle



def feature_counter(filepath = 'features/'):

    feature_files = os.listdir(filepath)

    for each in feature_files:
        f = open(filepath + each)
        features = pickle.load(f)
        f.close()
        if len(features) > 30:
            print each.split('.')[0].split('_')[1], len(features) 



if __name__ == '__main__':
    feature_counter()

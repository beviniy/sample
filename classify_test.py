#coding:utf-8


import cPickle as pickle
import json

from feature_extract import Room

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import shutil
import os
import random

from room_type import ROOM_TYPE_CHOICES

used_type = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]


def get_feature(intype='all'):
    path = 'features/feature_%s.pkl'
    
    f= open()
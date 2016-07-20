# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:50:04 2016

@author: Administrator
"""

import os
from room_type import ROOM_TYPE_CHOICES
import glob


def index(path = 'samples_pics/'):
    
    a = os.listdir(path)
    #print a
    os.chdir(path)
    print glob.glob('.*100.*')
    pass




if __name__ == '__main__':
    index()
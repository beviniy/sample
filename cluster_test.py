#coding:utf8

import cPickle as pickle
import json

from feature_extract import Room

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shutil
import os
import random

from room_type import ROOM_TYPE_CHOICES

def classification(pre_y, info):
    basedir = './classification/%d'
    #source = u'./samples_pics/房间%d.png'
    source = u'samples_pics/sub/%s_%s_%s.png'

    func = lambda x:(ROOM_TYPE_CHOICES[int(x[0])][1],int(x[1]),int(x[2]))

    for each in set(pre_y):
        if not os.path.exists(basedir%each):
            os.mkdir(basedir%each)
    for i, each in enumerate(pre_y):
        shutil.copy(source% func(info[i]), basedir%each)


def colormap(nums):
    colors = []
    for each in nums:
        colors.append((random.random(),random.random(),random.random()))
    return colors


def feature_fit(features, info):
    #features = feature_filter(features)
    X = np.array(features)
    X = StandardScaler().fit_transform(X)
    random_state = 170
    pre_y = KMeans(n_clusters=100, random_state=random_state).fit_predict(X)
    classification(pre_y, info)
    return
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')

    #pre_y
    #ax.scatter(X[:,0], X[:,1], X[:,2], c=(1,0.5,0),label=pre_y)
    #print type(pre_y),type(pre_y[np.array([1,2,3,4])])
    sy = set(pre_y)
    colors = colormap(sy)
    for each in set(sy):
        index = np.where(pre_y==each)[0]
        #print index
        #print pre_y[index]
        ax.scatter(X[index,0], X[index,1], X[index,2], c=colors[each], label=each)
    #colors
    #
    plt.legend()
    plt.show()


def feature_filter(features):
    diff = lambda a,i:max(a[:,i]) - min(a[:,i])
    for each in range(features.shape[1]):
        features[:,each] = (features[:,each]-min(features[:,each]))/diff(features,each) * 100

    #features[:,1] = features[:,1] * 10000
    return features



def get_feature(fname = 'taged_sample.pkl',fout = 'features/feature_%s.pkl',room_type='all'):

    f = open(fname,'rb')
    rooms = pickle.load(f)
    f.close()
    features = []

    for i,room in enumerate(rooms):
        #if len(room['points']) <4:
        #    print room['points']
        #    continue
        if room_type == 'all' or room_type == room['room_type']:
            roomp = Room(room['room_type'],room['intype_index'],room['tag'],*room['room_points'])
            features.append(roomp.extract())
        #print i,

    features = np.array(features, dtype='float')
    if not len(features):
        print '样本数为0'
    #else:
    #x,y,z = zip(*features)
    #    diff = lambda a,i:max(a[:,i]) - min(a[:,i])
    #    print diff(features,0),diff(features,1),diff(features,2)
    #return features
    f = open(fout % room_type ,'wb')
    pickle.dump(features,f)
    f.close()

def train(room_type, refresh = False):
    
    outpath = 'features/feature_%s.pkl'
    if refresh or (not os.path.exists(outpath%room_type)):
        print 'get feature for %s' % room_type
        get_feature(fout = outpath, room_type = room_type)
        
    #return
    f = open(outpath % room_type)
    features = pickle.load(f)
    f.close()
    
    print len(features)
    if len(features)>10:
        feature_fit(features[:,:5], features[:,-3:])
    else:
        print u'样本太少,%s:%s' %(room_type,len(features))

if __name__ == '__main__':
    #for each in zip(*ROOM_TYPE_CHOICES)[0]:
    #    print '-----------'
    #    print each
    #    train(each, 1)
    #train(3)
    train('all',0)

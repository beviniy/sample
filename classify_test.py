#coding:utf-8


import cPickle as pickle

from feature_extract import Room
import itertools
import numpy as np

import os
import sys

from room_type import ROOM_TYPE_CHOICES
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#out = open('result3.txt', 'w')
out = sys.stdout

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
         
classifiers = [
    KNeighborsClassifier(9),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

#used_type = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]

used_type = [1,2,3,4, 5 ,6, 7,8,10,12]

def get_data(features_index,intype='all'):
    path = 'features/feature_%s.pkl'

    f= open(path % intype)
    features = pickle.load(f)
    f.close()

    features = np.array([each for each in features if each[5] in used_type])
    #index = np.where(features[:,4].all() in used_type)

    return features[:,features_index], features[:,-3]


def train(features_index):
    X, y = get_data(features_index)
    #scaler = StandardScaler().fit(X)
    #print scaler.mean_, scaler.var_ ,scaler.scale_
    #X = scaler.transform(X)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=.1)
    '''
    from collections import Counter
    print Counter(y_train)
    print Counter(y_test)
    '''
    for name, clf in zip(names, classifiers):

        clf.fit(X_train, y_train)
        '''
        count = 0
        for i,each in enumerate(X_test):
            if clf.predict(each) == y_test[i]:
                count += 1
        score = float(count) / len(y_test)
        '''
        score = clf.score(X_test, y_test)
        out.write('name:%s\tscore:%s\n' % (name,score))
    pass
'''
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


def feature_train(room_type,features, refresh = False):
    outpath = 'features/feature_%s.pkl'
    if refresh or (not os.path.exists(outpath%room_type)):
        print 'get feature for %s' % room_type
        get_feature(fout = outpath, room_type = room_type)
    return
    #f = open(outpath % room_type)
    #features = pickle.load(f)
    #f.close()
    #if len(features)>10:
    #    feature_fit(features[:,:5], features[:,-3:])
    #else:
    #    print u'样本太少,%s:%s' %(room_type,len(features))
'''

if __name__ == '__main__':
    n_features = 5
    index = range(n_features)
    for each in index:
        for each_sub in itertools.combinations(index,each+1):
            out.write('----------------\n')
            out.write('%s\n' %'\t'.join((str(i) for i in each_sub)))
            train(each_sub)
            out.write('----------------\n')



out.close()

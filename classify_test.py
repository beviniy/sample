#coding:utf-8


import cPickle as pickle
#import json

#from feature_extract import Room
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.cluster import KMeans
#import shutil
#import os
#import random

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
def get_data(intype='all'):
    path = 'features/feature_%s.pkl'
    
    f= open(path % intype)
    features = pickle.load(f)
    f.close()
    
    features = np.array([each for each in features if each[5] in used_type])
    #index = np.where(features[:,4].all() in used_type)
    
    return features[:,:5], features[:,5]
    

def train():
    X, y = get_data()
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=.1)
    
    for name, clf in zip(names, classifiers):

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print 'name:%s\tscore:%s'%(name,score)
    pass

if __name__ == '__main__':
    train()
    
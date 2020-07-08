from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]

# create a clasifier
clf = svm.SVC()
clf.fit(X, y)

clf.predict([[2., 2.]])

#*******************************************************************************

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.svm import SVC
clf = SVC(kernel="linear")

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
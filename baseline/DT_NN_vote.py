__author__ = 'zhengyiwang'
import numpy as np
import math
import time
import os
from collections import Counter
from validation import build_load_model
from sklearn.neighbors.nearest_centroid import NearestCentroid
# from sklearn.neural_network import MLPClassifie
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# load data
start = time.time()
filename="data.csv"
X = np.loadtxt(filename,delimiter=',')
print 'load finished took %d s' % (time.time() - start)

# split training and testing data
y = X[:,-1]
X = X[:,:-1]
m = X.shape[0]
split = int(math.floor(m*0.8))
X_tr=X[:split]
X_te=X[split:int(split+split*0.4)]
y_tr=y[:split]
y_te=y[split:int(split+split*0.4)]
print X_tr.shape, X_te.shape

# training data
start = time.time()
clf = DecisionTreeClassifier(max_depth=10)

clf.fit(X_tr,y_tr)
print 'train finished took %d s' % (time.time() - start)

y_p =clf.predict(X_te)
accuracy = (y_p == y_te).sum()/float(len(y_te))
print 'testing accuracy =', accuracy

# validation dataset
root_folder = 'feature_txt'
categories = os.listdir(root_folder)
print categories
type_to_class = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock'
}
count = 0
total = 0
model = build_load_model()
for cat in categories:
    subfolder = os.path.join(root_folder,cat)
    if not os.path.isdir(subfolder) or cat.startswith('.'):
        continue
    print cat + ':'
    for file in os.listdir(subfolder):
        if not file.startswith('.') and int(file[-6:-4]) >= 80:
            X = np.loadtxt(os.path.join(subfolder, file),delimiter=',')
            y_p =clf.predict(X)
            # NN model loaded
            y_nn = model.predict(X)
            y_nn = [np.argmax(y) for y in y_nn]
            counter = Counter(y_p) + Counter(y_nn)
            genre = type_to_class[int(counter.most_common(1)[0][0])]
            print file + '...'+ genre
            total += 1
            if genre == cat:
                print "correct!"
                count += 1

print "validation accuracy = " + str(count *1.0/total)
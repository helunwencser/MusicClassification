__author__ = 'zhengyiwang'
import numpy as np
import math
import time
from sklearn.svm import SVC

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
X_te=X[split:]
y_tr=y[:split]
y_te=y[split:]
print X_tr.shape, X_te.shape

# training data
start = time.time()
cfl = SVC(C=1.0, kernel='rbf', degree=2, gamma=0.001,
            coef0=0.0, shrinking=True, probability=False,
            tol=0.001, cache_size=200, class_weight=None,
            verbose=True, max_iter=-1, decision_function_shape=None, random_state=None)

cfl.fit(X_tr,y_tr)
print 'train finished took %d s' % (time.time() - start)

y_p = cfl.predict(X_te)
accuracy = (y_p == y_te).sum()/float(len(y_te))
print 'accuracy =', accuracy

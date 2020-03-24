import numpy as np
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, normalize, Normalizer
from sklearn.metrics import explained_variance_score
from random import randint
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

mat = loadmat("Project Datasets/thyroidTrain.mat")
fig = plt.figure()

Xtrain = mat["trainThyroidInput"]
Ytrain = mat["trainThyroidOutput"]
Xtest = mat["testThyroidInput"] 

#print('Xtrain:\n', Xtrain)
print(f'Shape of Xtrain: {Xtrain.shape}')
print(f'Shape of Ytrain: {Ytrain.shape}')
print(f'Shape of Xtest: {Xtest.shape}')

Xtrain, Xval, Ytrain, Yval = train_test_split(
    Xtrain, Ytrain, test_size=0.2, random_state=42) 

Xtrain_normalized = normalize(Xtrain)
Xtest_normalized = normalize(Xtest)

pca = PCA()

Xtrain_reduced = pca.fit_transform(Xtrain_normalized)
Xtest_reduced = pca.fit_transform(Xtest_normalized)

print(f'Shape of transformed Xtrain: {Xtrain_reduced.shape}')
print(f'Shape of transformed Xtest: {Xtest_reduced.shape}')

fig, ax = plt.subplots()
#ax.scatter(Xtrain_reduced[:,0],Ytrain)
ax.scatter(Xtrain_reduced[:,0],Xtrain_reduced[:,1])
ax.set_ylabel('Valve Process')
ax.set_xlabel('Time')
plt.legend()
plt.show()
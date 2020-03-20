import numpy as np
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, train_test_split
import matplotlib.pylab as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, normalize, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
from sklearn.neural_network import MLPRegressor
from random import randint

mat = loadmat("Project Datasets/ChemTrainNew.mat")
fig = plt.figure()

Xtrain = mat["XtrainDS"]
Ytrain = mat["YtrainDS"]
Xtest = mat["XtestDS"]

#print('Xtrain:\n', Xtrain)
print(f'Shape of Xtrain: {Xtrain.shape}')
print(f'Shape of Ytrain: {Ytrain.shape}')
print(f'Shape of Xtest: {Xtest.shape}')

Xtrain_normalized = normalize(Xtrain)
Xtest_normalized = normalize(Xtest)

pca = PCA(n_components=2)

Xtrain_reduced = pca.fit_transform(Xtrain_normalized)
Xtest_reduced = pca.fit_transform(Xtest_normalized)

print(f'Shape of transformed Xtrain: {Xtrain_reduced.shape}')
print(f'Shape of transformed Xtest: {Xtest_reduced.shape}')

fig, ax = plt.subplots()
ax.scatter(Xtrain_reduced,Ytrain, label="True Price")
ax.set_ylabel('Valve Opening')
ax.set_xlabel('The process')
plt.legend()
fig.show()


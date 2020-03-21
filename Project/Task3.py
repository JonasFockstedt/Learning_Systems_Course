import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

mat = loadmat("Project Datasets/PowerTrainData.mat")

X_power_train = mat["powerTrainInput"].T
X_power_test = mat["powerTestInput"]
X_date_train = mat["powerTrainDate"]
Y_power_train = mat["powerTrainOutput"].T

print(X_power_train.shape)
print(X_power_test.shape)
print(X_date_train.shape)
print(Y_power_train.shape)

print(X_power_train)

pca = PCA(n_components=2)
X_power_train_reduced = pca.fit_transform(X_power_train)

fig, ax = plt.subplots()

ax.scatter(X_power_train_reduced[:,0], X_power_train_reduced[:,1], c=Y_power_train, edgecolor='black')

plt.show()
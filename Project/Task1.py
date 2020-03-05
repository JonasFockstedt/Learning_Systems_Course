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
from sklearn.metrics import explained_variance_score

mat = loadmat("Project Datasets/cnDieselTrain.mat")
fig, ax = plt.subplots()

# Load data.
Xtest = mat["cnTestX"].T
Xtrain = mat["cnTrainX"].T
Ytrain = mat["cnTrainY"][0]

print(f'Shape of Xtrain: {Xtrain.shape}')
print(f'Shape of Xtest: {Xtest.shape}')
print(f'Shape of Ytrain: {Ytrain.shape}')

# Xval and Yval will be used as testing.
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.2)

Xtrain_normalized = normalize(Xtrain)

#pca = KernelPCA(n_components=2, kernel='cosine').fit(Xtrain_normalized)
pca = PCA()

Xtrain_reduced = pca.fit_transform(Xtrain)
Xtest_reduced = pca.fit_transform(Xtest)
#n = len(Xtrain_reduced)

kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)

print(f'Shape of transformed Xtrain: {Xtrain_reduced.shape}')
print(f'Shape of transformed Xtest: {Xtest_reduced.shape}')

#print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))

regr = LinearRegression(n_jobs=-1)
svr = SVR()

lowest_mse = 0
regression_models = [LinearRegression(n_jobs=-1), SVR()]
mse = []

for number_of_components in range(1, 50):
    pca = PCA(n_components=number_of_components)
    Xtrain_reduced = pca.fit_transform(Xtrain)
    score = -1*cross_val_score(regr, Xtrain_reduced, Ytrain.ravel(),
                               cv=kf_10, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    mse.append(score)
    if lowest_mse == 0:
        lowest_mse = score
    elif score < lowest_mse:
        lowest_mse = score
        optimal_components = number_of_components

print(
    f'Number of components with least MSE: {optimal_components}\nLowest MSE: {lowest_mse}')
ax.plot(np.arange(1, 50, 1), mse, '-o', color='blue')
ax.set_ylabel('MSE')
ax.set_xlabel('Number of PCAs')
ax.set_title('MSE by number of PCAs')
plt.show()

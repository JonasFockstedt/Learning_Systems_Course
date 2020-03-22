import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from random import randint

mat = loadmat("Project Datasets/PowerTrainData.mat")

X_power_train = mat["powerTrainInput"].T
X_power_test = mat["powerTestInput"].T
X_date_train = mat["powerTrainDate"].T
Y_power_train = mat["powerTrainOutput"].T

print(f'Shape of Xtrain: {X_power_train.shape}')
print(f'Shape of Xtest: {X_power_test.shape}')
print(f'Shape of Xdate: {X_date_train.shape}')
print(f'Shape of Ytrain: {Y_power_train.shape}')

X_power_train_normalized = normalize(X_power_train)
X_power_train_scaled = MinMaxScaler().fit_transform(X_power_train)

pca = PCA(n_components=2)
X_power_train_reduced = pca.fit_transform(X_power_train_scaled)

models = {'SVR': SVR()}

def trainModels():
    rndm_state = randint(1,100)
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=rndm_state)
    for model in models.values():
        score = cross_val_score(model, X_power_train_normalized, Y_power_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error', n_jobs=1).mean()
        print(score)


def parameterTuning():
    parametersSVR = {'kernel': ('linear', 'rbf'), 'C': np.arange(1, 10, 1), 'epsilon': np.arange(0, 1, 0.1)}
    regr = GridSearchCV(models['SVR'], parametersSVR, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10)
    regr.fit(X_power_train_normalized, Y_power_train.ravel())
    print(regr.best_estimator_)
    print(regr.best_score_)

    
fig, ax = plt.subplots()

ax.scatter(X_power_train_scaled[:,0], X_power_train_scaled[:,1], edgecolor='black')
plt.title('Measurements of first two days')
plt.show()

if __name__ == '__main__':
    trainModels()
    parameterTuning()
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

mat = loadmat("Project Datasets/cnDieselTrain.mat")
fig = plt.figure()

# Load data.
Xtest = mat["cnTestX"].T
Xtrain = mat["cnTrainX"].T
Ytrain = mat["cnTrainY"][0]

print(f'Shape of Xtrain: {Xtrain.shape}')
print(f'Shape of Xtest: {Xtest.shape}')
print(f'Shape of Ytrain: {Ytrain.shape}')

# Xval and Yval will be used as testing.
Xtrain, Xval, Ytrain, Yval = train_test_split(
    Xtrain, Ytrain, test_size=0.2, random_state=42)

Xtrain_normalized = normalize(Xtrain)

#pca = KernelPCA(n_components=2, kernel='cosine').fit(Xtrain_normalized)
pca = PCA()

Xtrain_reduced = pca.fit_transform(Xtrain)
Xtest_reduced = pca.fit_transform(Xtest)
#n = len(Xtrain_reduced)

print(f'Shape of transformed Xtrain: {Xtrain_reduced.shape}')
print(f'Shape of transformed Xtest: {Xtest_reduced.shape}')

#print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))


def findOptimalNumberOfFeatures():
    lowest_mse = 0
    regression_models = [LinearRegression(
        n_jobs=-1), SVR(), DecisionTreeRegressor(criterion='mse', splitter="best"), MLPRegressor()]
    optimal_PCA_numbers = dict()
    mse = []
    standard_deviations = []
    figures = []

    for fig_number, model in enumerate(regression_models):
        # Things to reset for every model.
        kf_10 = KFold(n_splits=10, shuffle=True, random_state=randint(0, 100))
        optimal_PCA_numbers.update({type(model).__name__: 0})
        mse, standard_deviations = [], []
        ax = fig.add_subplot(4, 1, fig_number+1)
        optimal_components = 0
        lowest_mse = 0

        for number_of_components in range(1, 100):
            # Update components for PCA.
            pca = PCA(n_components=number_of_components)
            Xtrain_reduced = pca.fit_transform(Xtrain)
            score = -1*cross_val_score(model, Xtrain_reduced, Ytrain.ravel(),
                                       cv=kf_10, scoring='neg_mean_squared_error', n_jobs=-1).mean()
            mse.append(score)
            standard_deviations.append(np.std(mse))
            if lowest_mse == 0:
                lowest_mse = score
            elif score < lowest_mse:
                lowest_mse = score
                optimal_components = number_of_components
                optimal_PCA_numbers.update(
                    {type(model).__name__: optimal_components})

        mse = np.array(mse)
        standard_deviations = np.array(standard_deviations)
        ax.plot(np.arange(1, 100, 1), mse, marker='.', color='blue',
                label=f'Optimal PCAs: {optimal_components} at  MSE of {lowest_mse}.')
        ax.fill_between(np.arange(1, 100, 1), mse + standard_deviations, mse-standard_deviations,
                        alpha=0.5, facecolor='blue')
        ax.set_ylabel('MSE')
        ax.title.set_text(f'{type(model).__name__}')
        ax.legend()

    plt.xlabel('Number of PCAs')

    plt.show()


if __name__ == '__main__':
    findOptimalNumberOfFeatures()

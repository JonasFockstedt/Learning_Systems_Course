import numpy as np
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, train_test_split, GridSearchCV
import matplotlib.pylab as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, normalize, Normalizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
from sklearn.neural_network import MLPRegressor
from random import randint
from sklearn.neighbors import KNeighborsRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

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

# Below normalization gives a slightly worse MSE.
#transformer = Normalizer()
#Xtrain_normalized = transformer.transform(Xtrain)
#Xtest_normalized = transformer.transform(Xtest)

# Normalize data.
Xtrain_normalized = normalize(Xtrain)
Xtest_normalized = normalize(Xtest)

#pca = KernelPCA(n_components=2, kernel='cosine').fit(Xtrain_normalized)
pca = PCA()

Xtrain_reduced = pca.fit_transform(Xtrain)
Xtest_reduced = pca.fit_transform(Xtest)

print(f'Shape of transformed Xtrain: {Xtrain_reduced.shape}')
print(f'Shape of transformed Xtest: {Xtest_reduced.shape}')

optimal_PCA_numbers = dict()
grid_searches = dict()
regression_models = {'LinearRegression': LinearRegression(n_jobs=-1), 'SVR': SVR(), 'DecisionTreeRegressor': DecisionTreeRegressor(), 'MLPRegressor': MLPRegressor(), 'KNeighborsRegressor': KNeighborsRegressor(n_jobs=-1), 'Ridge': Ridge()}



def findOptimalNumberOfFeatures():
    lowest_mse = 0
    mse = []
    standard_deviations = []
    
    fig.subplots_adjust(hspace=.5)

    for fig_number, model in enumerate(regression_models.keys(), 1):
        print(f'Running {type(regression_models[model]).__name__} model...')
        # Things to set up for every model.
        kf_10 = KFold(n_splits=10, shuffle=True, random_state=randint(0, 100))
        optimal_PCA_numbers.update({type(regression_models[model]).__name__: 0})
        mse, standard_deviations = [], []
        optimal_components = 0
        lowest_mse = 0
        # Where to place the subplot.
        ax = fig.add_subplot(3, 2, fig_number)

        for number_of_components in range(1, 100):
            # Update components for PCA.
            pca = PCA(n_components=number_of_components)
            Xtrain_reduced = pca.fit_transform(Xtrain_normalized)
            # Run 10-fold cross validation.
            score = -1*cross_val_score(regression_models[model], Xtrain_reduced, Ytrain.ravel(),
                                       cv=kf_10, scoring='neg_mean_squared_error', n_jobs=-1).mean()
            mse.append(score)
            standard_deviations.append(np.std(mse))

            # If first run, assign current score as lowest MSE.
            if lowest_mse == 0:
                lowest_mse = score
            # If a new lowest MSE score is reached, save number of features.
            elif score < lowest_mse:
                lowest_mse = score
                optimal_components = number_of_components
                optimal_PCA_numbers.update(
                    {type(regression_models[model]).__name__: optimal_components})

        # Convert lists into arrays.
        mse = np.array(mse)
        standard_deviations = np.array(standard_deviations)
        # Plots the MSE.
        ax.plot(np.arange(1, 100, 1), mse, marker='.', color='blue',
                label=f'Optimal PCAs: {optimal_components} at MSE of {lowest_mse}.')
        # Plots standard deviation with half opacity.
        ax.fill_between(np.arange(1, 100, 1), mse + standard_deviations, mse - standard_deviations,
                        alpha=0.5, facecolor='blue')
        ax.set_ylabel('MSE')
        ax.title.set_text(f'{type(regression_models[model]).__name__}')
        ax.legend()

    plt.xlabel('Number of PCAs')
    plt.show(block=False)


def parameterTuning():
    # Pool of test parameters for each model.
    parametersLR = {'fit_intercept': ('True', 'False')}
    parametersSVR = {'kernel': ('linear', 'rbf'), 'C': np.arange(
        1, 10, 1), 'epsilon': np.arange(0, 1, 0.1)}
    parametersDT = {'criterion': ('mse', 'friedman_mse'), 'splitter': ('best', 'random')}
    parametersMLP = {'activation': ('identity', 'logistic')}
    parametersKNN = {'n_neighbors': np.arange(1,10,1), 'p': [1,2]}
    parametersRidge = {'solver': ('auto', 'svd', 'sag')}

    # Grouping the parameters together.
    parameters = {'LinearRegression': parametersLR, 'SVR': parametersSVR, 'DecisionTreeRegressor': parametersDT, 'MLPRegressor': parametersMLP, 'KNeighborsRegressor': parametersKNN, 'Ridge': parametersRidge}
    for model in regression_models.keys():
        print(f'Tuning {type(regression_models[model]).__name__}...')
        # Transform Xtrain based on the optimal number of features calculated previously.
        pca = PCA(n_components=optimal_PCA_numbers[model])
        Xtrain_reduced = pca.fit_transform(Xtrain_normalized)
        # Perform grid search.
        regr = GridSearchCV(regression_models[model], parameters[model], n_jobs=-1)
        regr.fit(Xtrain_reduced, Ytrain)
        # Add best parameter combination to dictionary.
        grid_searches.update({type(regression_models[model]).__name__: regr.best_estimator_})


if __name__ == '__main__':
    print('***FINDING OPTIMAL NUMBER OF FEATURES FOR EACH MODEL...***')
    findOptimalNumberOfFeatures()
    print('***FINDING OPTIMAL PARAMETERS FOR THE MODELS...***')
    parameterTuning()
    plt.show()

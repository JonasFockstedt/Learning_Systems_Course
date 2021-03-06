import numpy as np
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
from sklearn.neural_network import MLPRegressor
from random import randint
from sklearn.neighbors import KNeighborsRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

mat = loadmat("Project Datasets/ChemTrainNew.mat")
fig = plt.figure()

Xtrain = mat["XtrainDS"]
Ytrain = mat["YtrainDS"]
Xtest = mat["XtestDS"]

# Remove first column (time)
Xtrain = np.delete(Xtrain, 0, 1)
Xtest = np.delete(Xtest, 0, 1)

#print('Xtrain:\n', Xtrain)
print(f'Shape of Xtrain: {Xtrain.shape}')
print(f'Shape of Ytrain: {Ytrain.shape}')
print(f'Shape of Xtest: {Xtest.shape}')

# Xval and Yval will be used as testing.
Xtrain, Xval, Ytrain, Yval = train_test_split(
    Xtrain, Ytrain, test_size=0.2, random_state=42)

Xtrain_normalized = normalize(Xtrain)
Xtest_normalized = normalize(Xtest)

optimal_PCA_numbers = dict()
grid_searches = dict()
cv_scores = dict()
regression_models = {'LinearRegression': LinearRegression(n_jobs=-1), 'SVR': SVR(),
                     'DecisionTreeRegressor': DecisionTreeRegressor(),
                     'MLPRegressor': MLPRegressor(), 'KNeighborsRegressor': KNeighborsRegressor(n_jobs=-1), 'Ridge': Ridge()}


def findOptimalNumberOfFeatures():
    lowest_mse = 0
    mse = []
    standard_deviations = []

    fig.subplots_adjust(hspace=.5)
    rndm_state = randint(0, 100)

    for fig_number, model in enumerate(regression_models.keys(), 1):
        print(
            f'Finding optimal number of features for {type(regression_models[model]).__name__} model...')
        # Things to set up for every model.
        kf_10 = KFold(n_splits=10, shuffle=True, random_state=rndm_state)
        optimal_PCA_numbers.update(
            {type(regression_models[model]).__name__: 0})
        mse, standard_deviations = [], []
        optimal_components = 0
        lowest_mse = 0
        # Where to place the subplot.
        ax = fig.add_subplot(3, 2, fig_number)

        for number_of_components in range(1, 65):
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
        ax.plot(np.arange(1, 65, 1), mse, marker='.', color='blue',
                label=f'Optimal PCAs: {optimal_components} at MSE of {lowest_mse}.')
        # Plots standard deviation with half opacity.
        ax.fill_between(np.arange(1, 65, 1), mse + standard_deviations, mse - standard_deviations,
                        alpha=0.5, facecolor='blue')
        ax.set_ylabel('MSE')
        ax.title.set_text(f'{type(regression_models[model]).__name__}')
        ax.legend()

    plt.xlabel('Number of PCAs')
    plt.show()


def parameterTuning():
    # Pool of test parameters for each model.
    parametersLR = {'fit_intercept': ('True', 'False')}
    parametersSVR = {'kernel': ('linear', 'rbf'), 'C': np.arange(
        1, 10, 1), 'epsilon': np.arange(0, 1, 0.1)}
    parametersDT = {'criterion': (
        'mse', 'friedman_mse'), 'splitter': ('best', 'random')}
    parametersMLP = {'activation': ('identity', 'logistic')}
    parametersKNN = {'n_neighbors': np.arange(1, 10, 1), 'p': [1, 2]}
    parametersRidge = {'solver': ('auto', 'svd', 'sag')}

    # Grouping the parameters together.
    parameters = {'LinearRegression': parametersLR, 'SVR': parametersSVR, 'DecisionTreeRegressor': parametersDT,
                  'MLPRegressor': parametersMLP, 'KNeighborsRegressor': parametersKNN, 'Ridge': parametersRidge}
    for model in regression_models.keys():
        print(f'Tuning {type(regression_models[model]).__name__}...')
        # Transform Xtrain based on the optimal number of features calculated previously.
        pca = PCA(n_components=optimal_PCA_numbers[model])
        Xtrain_reduced = pca.fit_transform(Xtrain_normalized)
        # Perform grid search.
        regr = GridSearchCV(regression_models[model], parameters[model],
                            scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10)
        regr.fit(Xtrain_reduced, Ytrain.ravel())
        # Add best parameter combination to dictionary.
        grid_searches.update(
            {type(regression_models[model]).__name__: regr.best_estimator_})
        # Add score of best parameters to dictionary
        cv_scores.update(
            {type(regression_models[model]).__name__: regr.best_score_})


def plotCVScores():
    plt.ylabel('Mean squared error regression loss (higher is better)')
    plt.title('Cross-validation scores of regression models')
    bars = plt.bar(list(cv_scores.keys()), list(
        cv_scores.values()), align='center')
    # Add value above every bar.
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 0.05, yval)
    plt.show()


def trainModel():
    # Fetch best performing model.
    the_model = grid_searches[max(cv_scores, key=cv_scores.get)]
    # Fetch the number of optimal features for the model.
    pca = PCA(n_components=optimal_PCA_numbers[str(type(the_model).__name__)])

    Xval_normalized = normalize(Xval)
    Xval_reduced = pca.fit_transform(Xval_normalized)
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
    # Run 10-fold cross validation.
    score = cross_val_score(the_model, Xval_reduced, Yval.ravel(),
                            cv=kf_10, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    print(
        f'Mean validation score of the {type(the_model).__name__} model: {score} (MSE).')

    Xtest_reduced = pca.fit_transform(Xtest_normalized)
    predictions = the_model.predict(Xtest_reduced)

    Xtest_to_visualize = PCA(n_components=2).fit_transform(Xtest)

    print(f'Best model: \n{the_model}')
    print(f'Predictions: \n{predictions}')

    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.title.set_text('Input of test data')
    ax1.set_ylabel('Principal component 1')
    ax1.set_xlabel('Measurement (30min in between)')
    ax1.plot(Xtest_to_visualize[:, 0], color='blue')

    ax2.title.set_text('Predicted output of test data')
    ax2.set_ylabel('Valve opening')
    ax2.set_xlabel('Measurement (30min in between)')
    ax2.plot(predictions, color='red')

    plt.show()


if __name__ == '__main__':
    print('***FINDING OPTIMAL NUMBER OF FEATURES FOR EACH MODEL...***')
    findOptimalNumberOfFeatures()
    print('***FINDING OPTIMAL PARAMETERS FOR THE MODELS...***')
    parameterTuning()
    plotCVScores()
    trainModel()

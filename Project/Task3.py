import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from random import randint
import timeit


start_time = timeit.default_timer()
mat = loadmat("Project Datasets/PowerTrainData.mat")

X_power_train = mat["powerTrainInput"].T
X_power_test = mat["powerTestInput"].T
X_date_train = mat["powerTrainDate"].T
Y_power_train = mat["powerTrainOutput"].T

print(f'Shape of Xtrain: {X_power_train.shape}')
print(f'Shape of Xtest: {X_power_test.shape}')
print(f'Shape of Xdate: {X_date_train.shape}')
print(f'Shape of Ytrain: {Y_power_train.shape}')

# Split training data.
X_power_train, X_power_val, Y_power_train, Y_power_val = train_test_split(
    X_power_train, Y_power_train, test_size=0.2, random_state=42)

X_power_train_normalized = normalize(X_power_train)


regression_models = {'SVR': SVR(), 'SGDRegressor': SGDRegressor(),
                     'DecisionTreeRegressor': DecisionTreeRegressor(), 'RandomForestRegressor': RandomForestRegressor(n_jobs=-1),
                     'KNeighborsRegressor': KNeighborsRegressor(n_jobs=-1), 'MLPRegressor': MLPRegressor()}
optimal_PCA_numbers = dict()
mse_scores = {'SVR': [], 'SGDRegressor': [], 'DecisionTreeRegressor':
              [], 'RandomForestRegressor': [], 'KNeighborsRegressor': [], 'MLPRegressor': []}
standard_deviation_scores = {'SVR': [], 'SGDRegressor': [],
                             'DecisionTreeRegressor': [], 'RandomForestRegressor': [], 'KNeighborsRegressor': [], 'MLPRegressor': []}
grid_searches = dict()
cv_scores = dict()


def findOptimalNumberOfFeatures():
    print('***FINDING OPTIMAL NUMBER OF FEATURES FOR EACH MODEL***')
    rndm_state = randint(0, 100)

    for model in regression_models.keys():
        print(
            f'Finding optimal number of features for {type(regression_models[model]).__name__} model...')
        # Things to set up for every model.
        kf_10 = KFold(n_splits=10, shuffle=True, random_state=rndm_state)
        optimal_PCA_numbers.update(
            {type(regression_models[model]).__name__: 0})
        optimal_components = 0
        lowest_mse = 0

        # Begin with only 1 feature, then add one feature until all features have been tested.
        for number_of_components in range(1, X_power_train.shape[1] + 1):
            # Update components for PCA.
            pca = PCA(n_components=number_of_components)
            Xtrain_reduced = pca.fit_transform(X_power_train_normalized)
            # Run 10-fold cross validation.
            score = -1*cross_val_score(regression_models[model], Xtrain_reduced, Y_power_train.ravel(),
                                       cv=kf_10, scoring='neg_mean_squared_error', n_jobs=-1).mean()

            mse_scores[model].append(score)
            standard_deviation_scores[model].append(np.std(mse_scores[model]))

            # If first run, assign current score as lowest MSE.
            if lowest_mse == 0:
                lowest_mse = score
                optimal_components = number_of_components
                optimal_PCA_numbers.update(
                    {type(regression_models[model]).__name__: optimal_components})
            # If a new lowest MSE score is reached, save number of features.
            elif score < lowest_mse:
                lowest_mse = score
                optimal_components = number_of_components
                optimal_PCA_numbers.update(
                    {type(regression_models[model]).__name__: optimal_components})


def plotMSEScores():
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)

    for fig_number, model in enumerate(regression_models.keys(), 1):
        ax = fig.add_subplot(3, 2, fig_number)
        mse = np.array(mse_scores[model])
        standard_deviations = np.array(standard_deviation_scores[model])

        # Plots the MSE.
        ax.plot(np.arange(1, X_power_train.shape[1]+1, 1), mse_scores[model], marker='.', color='blue',
                label=f'Optimal PCAs: {optimal_PCA_numbers[model]} at MSE of {min(mse)}.')
        # Plots standard deviation with half opacity.
        ax.fill_between(np.arange(1, X_power_train.shape[1] + 1, 1), mse + standard_deviations, mse - standard_deviations,
                        alpha=0.5, facecolor='blue')
        ax.set_ylabel('MSE')
        ax.title.set_text(f'{type(regression_models[model]).__name__}')
        ax.legend()

    plt.xlabel('Number of PCAs')
    plt.show()


def parameterTuning():
    print('***TUNING PARAMETERS...***')
    parametersSVR = {'gamma': (
        'scale', 'auto'), 'C': np.arange(1, 10, 1), 'epsilon': np.arange(0, 10, 0.1)}
    parametersSGD = {'alpha': np.arange(0.00001, 0.005, 0.0005), 'max_iter': np.arange(
        1000, 5000, 500), 'epsilon': np.arange(0.01, 1, 0.05)}
    parametersDT = {'criterion': (
        'mse', 'friedman_mse', 'mae'), 'splitter': ('best', 'random')}
    parametersRF = {'n_estimators': np.arange(50, 200, 50)}
    parametersKNN = {'n_neighbors': np.arange(
        1, 8, 1), 'weights': ('uniform', 'distance'), 'p': (1, 2)}
    parametersMLP = {'solver': ('lbfgs', 'sgd', 'adam'),
                     'alpha': np.arange(0.00005, 0.001, 0.0005)}

    parameters = {'SVR': parametersSVR, 'SGDRegressor': parametersSGD, 'DecisionTreeRegressor': parametersDT,
                  'RandomForestRegressor': parametersRF, 'KNeighborsRegressor': parametersKNN, 'MLPRegressor': parametersMLP}
    for model in regression_models.keys():
        print(f'Tuning the {model} model...')
        # Perform grid search.
        regr = GridSearchCV(regression_models[model], parameters[model],
                            scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10)
        regr.fit(X_power_train_normalized, Y_power_train.ravel())
        print(regr.best_estimator_)
        print(regr.best_score_)
        # Add best parameter combination to dictionary.
        grid_searches.update({model: regr.best_estimator_})
        # Add score of best parameters to dictionary
        cv_scores.update({model: regr.best_score_})


def plotCVScores():
    plt.ylabel('Mean squared error regression loss')
    plt.title('Cross-validation scores of regression models')
    bars = plt.bar(list(cv_scores.keys()), list(
        cv_scores.values()), align='center')
    # Add value above every bar.
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 0.05, yval)
    plt.show()


def trainBestModel():
    # Fetch the best performing model.
    best_model = grid_searches[max(cv_scores, key=cv_scores.get)]

    # Normalize validation data.
    X_power_val_normalized = normalize(X_power_val)
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
    # Run 10-fold cross validation.
    score = -1*cross_val_score(best_model, X_power_val_normalized, Y_power_val.ravel(),
                               cv=kf_10, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    print(
        f'Average validation score of the {type(best_model).__name__} model: {score} (MSE).')

    X_power_test_normalized = normalize(X_power_test)
    predictions = best_model.predict(X_power_test_normalized)

    print(f'Best model: \n{best_model}')
    print(f'Predictions: \n{predictions}')

    plt.title(f'Predicted output based on {type(best_model).__name__}')
    plt.xlabel('Day')
    plt.ylabel('Power load (MW)')
    plt.plot(np.arange(0, len(X_power_test[:, 0]), 1),
             X_power_test[:, 0], color='blue', label='Power load today')
    plt.plot(np.arange(1, len(predictions)+1, 1), predictions,
             color='red', label='Predicted power load the next day')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    findOptimalNumberOfFeatures()
    plotMSEScores()
    parameterTuning()
    plotCVScores()
    trainBestModel()
    print(f'Runtime: {timeit.default_timer() - start_time}s')

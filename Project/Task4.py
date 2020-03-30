import warnings
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from random import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning,
                        module="sklearn")

mat = loadmat("Project Datasets/thyroidTrain.mat")

Xtrain = mat["trainThyroidInput"]
Ytrain = mat["trainThyroidOutput"]
Xtest = mat["testThyroidInput"]

# Split training data.
Xtrain, Xval, Ytrain, Yval = train_test_split(
    Xtrain, Ytrain, test_size=0.2, random_state=42)

#print('Xtrain:\n', Xtrain)
print(f'Shape of Xtrain: {Xtrain.shape}')
print(f'Shape of Ytrain: {Ytrain.shape}')
print(f'Shape of Xtest: {Xtest.shape}')

# Normalize the data.
Xtrain_normalized = normalize(Xtrain)
Xtest_normalized = normalize(Xtest)

optimal_PCA_numbers = dict()
grid_searches = dict()
cv_scores = dict()
classification_models = {'KNeighborsClassifier': KNeighborsClassifier(), 'DecisionTreeClassifier': DecisionTreeClassifier(),
                         'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
                         'MLPClassifier': MLPClassifier()}
acc_scores = {'KNeighborsClassifier': [],
              'DecisionTreeClassifier': [], 'RandomForestClassifier': [], 'MLPClassifier': []}
standard_deviation_scores = {'KNeighborsClassifier': [],
                             'DecisionTreeClassifier': [], 'RandomForestClassifier': [], 'MLPClassifier': []}


def findOptimalNumberOfFeatures():
    print('***FINDING OPTIMAL NUMBER OF FEATURES FOR EACH MODEL***')

    for model in classification_models.keys():
        print(
            f'Finding optimal number of features for {type(classification_models[model]).__name__} model...')
        # Things to set up for every model.
        kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
        optimal_PCA_numbers.update(
            {type(classification_models[model]).__name__: 0})
        optimal_components = 0
        highest_acc = 0

        # Begin with only 1 feature, then add one feature until all features have been tested.
        for number_of_components in range(1, Xtrain.shape[1] + 1):
            # Update components for PCA.
            pca = PCA(n_components=number_of_components)
            Xtrain_reduced = pca.fit_transform(Xtrain_normalized)
            # Run 10-fold cross validation.
            score = 100*cross_val_score(classification_models[model], Xtrain_reduced, Ytrain,
                                        cv=kf_10, scoring='accuracy', n_jobs=-1).mean()

            acc_scores[model].append(score)
            standard_deviation_scores[model].append(np.std(acc_scores[model]))

            # If first run, assign current score as highest accuracy.
            if highest_acc == 0:
                highest_acc = score
                optimal_components = number_of_components
                optimal_PCA_numbers.update(
                    {type(classification_models[model]).__name__: optimal_components})
            # If a new highest accuracy score is reached, save number of features.
            elif score > highest_acc:
                highest_acc = score
                optimal_components = number_of_components
                optimal_PCA_numbers.update(
                    {type(classification_models[model]).__name__: optimal_components})


def plotAccuracyScores():
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)

    for fig_number, model in enumerate(classification_models.keys(), 1):
        ax = fig.add_subplot(2, 2, fig_number)
        acc = np.array(acc_scores[model])
        standard_deviations = np.array(standard_deviation_scores[model])

        # Plots the MSE.
        ax.plot(np.arange(1, Xtrain.shape[1]+1, 1), acc_scores[model], marker='.', color='blue',
                label=f'Optimal PCAs: {optimal_PCA_numbers[model]} at accuracy of {max(acc)}%.')
        # Plots standard deviation with half opacity.
        ax.fill_between(np.arange(1, Xtrain.shape[1] + 1, 1), acc + standard_deviations, acc - standard_deviations,
                        alpha=0.5, facecolor='blue')
        ax.set_ylabel('Accuracy (%)')
        ax.title.set_text(f'{type(classification_models[model]).__name__}')
        ax.legend()

    plt.xlabel('Number of PCAs')
    plt.show()


def parameterTuning():
    parametersKNN = {'n_neighbors': np.arange(
        1, 10, 1), 'weights': ('uniform', 'distance'), 'p': (1, 2)}
    parametersSVC = {'C': np.arange(1, 15, 1), 'penalty': ('l1', 'l2')}
    parametersLR = {'penalty': ('l1', 'l2'), 'C': np.arange(
        1, 10, 1), 'solver': ('lbfgs', 'liblinear')}
    parametersDT = {'criterion': (
        'gini', 'entropy'), 'splitter': ('best', 'random')}
    parametersRF = {'n_estimators': np.arange(
        50, 200, 50), 'criterion': ('gini', 'entropy')}
    parametersMLP = {'solver': ('lbfgs', 'sgd', 'adam'),
                     'alpha': np.arange(0.00005, 0.001, 0.0005)}

    parameters = {'KNeighborsClassifier': parametersKNN, 'LinearSVC': parametersSVC, 'LogisticRegression': parametersLR, 'DecisionTreeClassifier': parametersDT,
                  'RandomForestClassifier': parametersRF,  'MLPClassifier': parametersMLP}

    for model in classification_models.keys():
        print(f'Tuning the {model} model...')
        # Perform grid search.
        classifier = GridSearchCV(
            classification_models[model], parameters[model], scoring='accuracy', n_jobs=-1, cv=10)
        classifier.fit(Xtrain_normalized, Ytrain)
        print(classifier.best_estimator_)
        print(classifier.best_score_)
        # Add best parameter combination to dictionary.
        grid_searches.update({model: classifier.best_estimator_})
        # Add score of best parameters to dictionary
        cv_scores.update({model: classifier.best_score_*100})


def plotCVScores():
    plt.ylabel('Accuracy (%)')
    plt.title('Cross-validation scores of classification models')
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
    Xval_normalized = normalize(Xval)
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
    # Run 10-fold cross validation.
    score = 100*cross_val_score(best_model, Xval_normalized,
                                Yval, cv=kf_10, scoring='accuracy', n_jobs=-1).mean()
    print(
        f'Average validation score of the {type(best_model).__name__} model: {score} (accuracy %).')

    Xtest_normalized = normalize(Xtest)
    predictions = best_model.predict(Xtest_normalized)
    X_test_visualize = PCA(n_components=2).fit_transform(Xtest_normalized)

    # Fetch the predictions.
    normal = X_test_visualize[predictions[:, 0] == 1]
    hypothyroid = X_test_visualize[predictions[:, 1] == 1]
    hyperthyroid = X_test_visualize[predictions[:, 2] == 1]

    plt.scatter(normal[:, 0], normal[:, 1], color='blue', label='Normal')
    plt.scatter(hypothyroid[:, 0], hypothyroid[:, 1],
                color='red', label='Hypothyroid')
    plt.scatter(hyperthyroid[:, 0], hyperthyroid[:, 1],
                color='green', label='Hyperthyroid')
    plt.title(
        f'Predicted output based on the {type(best_model).__name__} model')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    findOptimalNumberOfFeatures()
    plotAccuracyScores()
    parameterTuning()
    plotCVScores()
    trainBestModel()

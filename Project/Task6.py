import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import normalize

mat = loadmat('Project Datasets/ECGITtrain.mat')

X_train = mat['inputECGITtrain']
Y_train = mat['outputECGITtrain']
X_test = mat['inputECGITtest']


def extractImportantFeatures(the_array):
    first_array = the_array[:, 18:26].copy()
    second_array = the_array[:, 44:52].copy()
    third_array = the_array[:, 70:78].copy()
    fourth_array = the_array[:, 96:104].copy()
    fifth_array = the_array[:, 122:130].copy()
    sixth_array = the_array[:, 148:156].copy()
    seventh_array = the_array[:, 174:182].copy()
    eighth_array = the_array[:, 200:208].copy()
    ninth_array = the_array[:, 226:234].copy()
    tenth_array = the_array[:, 252:260].copy()
    eleventh_array = the_array[:, 278:286].copy()
    twelfth_array = the_array[:, 304:312].copy()

    return np.concatenate((first_array, second_array, third_array, fourth_array, fifth_array, sixth_array, seventh_array, eighth_array, ninth_array, tenth_array, eleventh_array, twelfth_array), axis=1)


# Extract the most important features, according to the description.
X_train = extractImportantFeatures(X_train)
X_test = extractImportantFeatures(X_test)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of Y_train: {Y_train.shape}')
print(f'Shape of X_test: {X_test.shape}')

# Split training data.
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42)

X_train_normalized = normalize(X_train)

classification_models = {'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1), 'DecisionTreeClassifier': DecisionTreeClassifier(),
                         'RandomForestClassifier': RandomForestClassifier(n_jobs=-1), 'SVC': SVC(),
                         'LogisticRegression': LogisticRegression(n_jobs=-1), 'MLPClassifier': MLPClassifier()}
optimal_PCA_numbers = dict()
grid_searches = dict()
cv_scores = dict()
acc_scores = {'KNeighborsClassifier': [], 'DecisionTreeClassifier': [],
              'RandomForestClassifier': [], 'SVC': [], 'LogisticRegression': [], 'MLPClassifier': []}
standard_deviation_scores = {'KNeighborsClassifier': [], 'DecisionTreeClassifier': [],
                             'RandomForestClassifier': [], 'SVC': [], 'LogisticRegression': [], 'MLPClassifier': []}


def findOptimalFeatures():
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
        for number_of_components in range(1, 96):
            # Update components for PCA.
            pca = PCA(n_components=number_of_components)
            Xtrain_reduced = pca.fit_transform(X_train_normalized)
            # Run 10-fold cross validation.
            score = 100*cross_val_score(classification_models[model], Xtrain_reduced, Y_train.ravel(),
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
        ax = fig.add_subplot(3, 2, fig_number)
        acc = np.array(acc_scores[model])
        standard_deviations = np.array(standard_deviation_scores[model])

        # Plots the MSE.
        ax.plot(np.arange(1, 96, 1), acc_scores[model], marker='.', color='blue',
                label=f'Optimal PCAs: {optimal_PCA_numbers[model]} at accuracy of {max(acc)}%.')
        # Plots standard deviation with half opacity.
        ax.fill_between(np.arange(1, 96, 1), acc + standard_deviations, acc - standard_deviations,
                        alpha=0.5, facecolor='blue')
        ax.set_ylabel('Accuracy (%)')
        ax.title.set_text(f'{type(classification_models[model]).__name__}')
        ax.legend()

    plt.xlabel('Number of PCAs')
    plt.show()


def parameterTuning():
    parametersKNN = {'n_neighbors': np.arange(
        1, 10, 1), 'weights': ('uniform', 'distance'), 'p': (1, 2)}
    parametersDT = {'criterion': (
        'gini', 'entropy'), 'splitter': ('best', 'random')}
    parametersRF = {'n_estimators': np.arange(
        50, 200, 50), 'criterion': ('gini', 'entropy')}
    parametersSVC = {'C': np.arange(1, 10, 1), 'kernel': ('linear', 'poly'),
                     'gamma': ('scale', 'auto')}
    parametersLR = {'penalty': ('l1', 'l2'), 'C': np.arange(1, 10, 1),
                    'solver': ('lbfgs', 'liblinear')}
    parametersMLP = {'solver': ('lbfgs', 'sgd', 'adam'),
                     'alpha': np.arange(0.00005, 0.001, 0.0005)}

    parameters = {'KNeighborsClassifier': parametersKNN, 'DecisionTreeClassifier': parametersDT,
                  'RandomForestClassifier': parametersRF, 'SVC': parametersSVC, 'LogisticRegression': parametersLR, 'MLPClassifier': parametersMLP}
    for model in classification_models.keys():
        print(f'Tuning the {model} model...')
        # Perform grid search.
        classifier = GridSearchCV(
            classification_models[model], parameters[model], scoring='accuracy', n_jobs=-1, cv=10)
        classifier.fit(X_train_normalized, Y_train.ravel())
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
    X_val_normalized = normalize(X_val)
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
    # Run 10-fold cross validation.
    score = 100*cross_val_score(best_model, X_val_normalized,
                                Y_val.ravel(), cv=kf_10, scoring='accuracy', n_jobs=-1).mean()
    print(
        f'Average validation score of the {type(best_model).__name__} model: {score} (accuracy %).')

    X_test_normalized = normalize(X_test)
    predictions = best_model.predict(X_test_normalized)

    print(f'Prediction: \n{predictions}')
    print(f'Best model: \n{best_model}')

    X_test_visualize = PCA(n_components=2).fit_transform(X_test)

    ti_pattern = X_test_visualize[predictions == 1]
    non_ti_pattern = X_test_visualize[predictions == 0]

    plt.scatter(ti_pattern[:, 0], ti_pattern[:, 1], color='red', label='TI')
    plt.scatter(non_ti_pattern[:, 0], non_ti_pattern[:,
                                                     1], color='blue', label='Not TI')

    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.title(f'Predictions based on the {type(best_model).__name__} model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    findOptimalFeatures()
    plotAccuracyScores()
    parameterTuning()
    plotCVScores()
    trainBestModel()
    pass

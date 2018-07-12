import pandas as pd
import numpy as np

import lightgbm as lgb

import csv

import random

N_FOLDS = 5
MAX_EVALS = 1000
OUT_FILE = '../progress/random_search_final_manual_features.csv'

def objective(hyperparameters, iteration):
    """Objective function for random search. Returns
       the cross validation score from a set of hyperparameters."""

    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']

    if 'silent' in hyperparameters.keys():
        del hyperparameters['silent']

     # Perform n_folds cross validation with early stopping
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS,
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)

    # Best score is last in cv results
    score = cv_results['auc-mean'][-1]

    # Number of estimators os length of results
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators

    # LightGBM adds these to hyperparameters but they are not necessary
    del hyperparameters['metric'], hyperparameters['verbose']

    return [score, hyperparameters, iteration]


def random_search(param_grid, max_evals = MAX_EVALS, start = 0):
    """Random search for hyperparameter tuning"""

    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))


    for i in range(start, MAX_EVALS):

        # Choose random hyperparameters
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        # Set correct subsample
        random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(random_params, i)
        results.loc[i, :] = eval_results

        # Write results to line of file
        of_connection = open(OUT_FILE, 'a')
        writer = csv.writer(of_connection)
        writer.writerow(eval_results)

        print('Iteration: {} Cross Validation ROC AUC: {:.5f}.'.format(
            i + 1, eval_results[0]))

    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)

    return results


if __name__ == "__main__":

    # Read in the data and extract labels/features
    print('Reading in data')
    features = pd.read_csv('../input/final_manual_features.csv')
    train = features[features['TARGET'].notnull()].copy()
    del features
    train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))
    train = train.drop(columns = ['TARGET', 'SK_ID_CURR'])

    # Hyperparameter grid
    param_grid = {
        'is_unbalance': [True, False],
        'boosting_type': ['gbdt', 'goss', 'dart'],
        'num_leaves': list(range(20, 150)),
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
        'subsample_for_bin': list(range(20000, 300000, 20000)),
        'min_child_samples': list(range(20, 500)),
        'reg_alpha': list(np.linspace(0, 1)),
        'reg_lambda': list(np.linspace(0, 1)),
        'colsample_bytree': list(np.linspace(0.6, 1, 10)),
        'subsample': list(np.linspace(0.5, 1, 100))
    }

    print('Creating train set')
    train_set = lgb.Dataset(train, train_labels)

    # Create a new file and write the column names
    headers = ["score", "hyperparameters", "iteration"]
    of_connection = open(OUT_FILE, 'w')
    writer = csv.writer(of_connection)
    writer.writerow(headers)
    of_connection.close()

    start = 0

    print('Starting Random Search for {} iterations from {}.'.format(MAX_EVALS,
                                     start))

    # Random search
    results = random_search(param_grid, MAX_EVALS, start)

    print('Saving results')
    results.to_csv('../progress/random_results_final_manual_features.csv', index = False)
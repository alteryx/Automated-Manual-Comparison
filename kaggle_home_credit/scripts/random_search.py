import pandas as pd
import numpy as np

import lightgbm as lgb

from timeit import default_timer as timer

import csv

import random

N_FOLDS = 5
MAX_EVALS = 100

def objective(hyperparameters, iteration):
    """Objective function for random search. Returns
       the cross validation score from a set of hyperparameters."""

    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']

    start = timer()
     # Perform n_folds cross validation with early stopping
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000,
                         nfold = N_FOLDS,
                        early_stopping_rounds = 100, metrics = 'auc')

    time = timer() - start
    # Best score is last in cv results
    score = cv_results['auc-mean'][-1]
    std = cv_results['auc-stdv'][-1]

    # Number of estimators os length of results
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators

    return [score, std, hyperparameters, iteration, time]


def random_search(param_grid, out_file, max_evals = MAX_EVALS):
    """Random search for hyperparameter tuning"""

    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'std', 'params', 'iteration', 'time'],
                                  index = list(range(MAX_EVALS)))


    for i in range(MAX_EVALS):

        # Choose random hyperparameters
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        # Set correct subsample
        random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(random_params, i)
        results.loc[i, :] = eval_results

        # Write results to line of file
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow(eval_results)

        # Make sure to close file
        of_connection.close()

        print('Iteration: {} Cross Validation ROC AUC: {:.5f}.'.format(
            i + 1, eval_results[0]))

    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)

    return results

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Input Features and Output File")
    
    parser.add_argument(action="store", dest = "in_file")
    parser.add_argument(action = "store", dest = "out_file")
    
    args = parser.parse_args()
    
    in_file = '../input/%s.csv' % args.in_file
    out_file = '../results/%s.csv' % args.out_file

    # Read in the data and extract labels/features
    print('Reading in data')
    features = pd.read_csv(in_file)
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
        'min_child_samples': list(range(20, 500, 5)),
        'reg_alpha': list(np.linspace(0, 1)),
        'reg_lambda': list(np.linspace(0, 1)),
        'colsample_bytree': list(np.linspace(0.6, 1, 10)),
        'subsample': list(np.linspace(0.5, 1, 100))
    }

    print('Creating train set')
    train_set = lgb.Dataset(train, train_labels)

    # Create a new file and write the column names
    headers = ["score", "std", "hyperparameters", "iteration", "time"]
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    writer.writerow(headers)
    of_connection.close()

    print('Starting Search')

    # Random search
    results = random_search(param_grid, out_file, MAX_EVALS)

    print('Saving results')
    results.to_csv('../results/%s_finished.csv' % out_file, index = False)

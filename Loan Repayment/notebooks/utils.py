# Data manipulation
import pandas as pd
import numpy as np

# modeling
import lightgbm as lgb

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluating dictionary
import ast

RSEED = 50

def format_data(features):
    """Format a set of training and testing features joined together
       into separate sets for machine learning"""
    
    train = features[features['TARGET'].notnull()].copy()
    test = features[features['TARGET'].isnull()].copy()
    
    train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))
    test_ids = list(test['SK_ID_CURR'])
    
    train = train.drop(columns = ['TARGET', 'SK_ID_CURR'])
    test = test.drop(columns = ['TARGET', 'SK_ID_CURR'])
    
    feature_names = list(train.columns)
    
    return train, train_labels, test, test_ids, feature_names


def cross_validate_and_submit(features, best_hyp):
    """Function for assessing cross validation, calculating feature importances,
    and making a submission dataframe.
    
    Args:
        features (dataframe): dataset
        
    Returns:
        cv_results (dict): cross validation results. Keys will be 'auc-mean' and 'auc-stdv'.
        
        fi (dataframe): feature importance dataframe. Columns are 'feature' and 'importance'
        
        submission (dataframe): dataframe that can be submitted to the competition.
    
    """
    train, train_labels, test, test_ids, feature_names = format_data(features)
    
    train_set = lgb.Dataset(train, label = train_labels)
    
    cv_results = lgb.cv(best_hyp, train_set = train_set, seed = RSEED, nfold = 5, metrics = 'auc',
                        early_stopping_rounds = 100, num_boost_round = 10000)
    
    estimators = len(cv_results['auc-mean'])
    print('5-Fold CV ROC AUC: {:.5f} with std: {:.5f}.'.format(
           cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
    
    model = lgb.LGBMClassifier(**best_hyp, n_estimators = estimators)
    model.fit(train, train_labels)
    
    fi = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    
    preds = model.predict_proba(test)[:, 1]
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
    
    return cv_results, fi, submission

def plot_feature_importances(df, n = 15, threshold = None):
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".
    
        n (int): Number of most important features to plot. Default is 15.
    
        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.
        
    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column
    
    Note:
    
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance
    
    """
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'blue', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    return df


def evaluate(fm, hyp_results):
    """Evaluate a feature matrix using the hyperparameter tuning results.
    
    Parameters:
        fm (dataframe): feature matrix with observations in the rows and features in the columns. This will
                        be passed to `format_data` and hence must have a train set where the `TARGET` values are 
                        not null and a test set where `TARGET` is null. Must also have the `SK_ID_CURR` column.
        
        hyp_results (dataframe): results from hyperparameter tuning. Must have column `score` (where higher is better)
                                 and `params` holding the model hyperparameters
                                 
    Returns:
        results (dataframe): the cross validation roc auc from the default hyperparameters and the 
                             optimal hyperparameters
        
        feature_importances (dataframe): feature importances from the gradient boosting machine. Columns are 
                                          `feature` and `importance`. This can be used in `plot_feature_importances`.
                                          
        submission (dataframe): Predictions which can be submitted to the Kaggle Home Credit competition. Save
                                these as `submission.to_csv("filename.csv", index = False)` and upload
       """
    
    print('Number of features: ', (fm.shape[1] - 2))
    
    # Format the feature matrix 
    train, train_labels, test, test_ids, feature_names = format_data(fm)
    
    # Training set 
    train_set = lgb.Dataset(train, label = train_labels)

    # Dataframe to hold results
    results = pd.DataFrame(columns = ['default_auc', 'default_auc_std', 
                                      'opt_auc', 'opt_auc_std', 
                                      'random_search_auc'], index = [0])

    # Create a default model and find the hyperparameters
    model = lgb.LGBMClassifier()
    default_hyp = model.get_params()
    
    # Remove n_estimators because this is found through early stopping
    del default_hyp['n_estimators'], default_hyp['silent']

    # Cross validation with default hyperparameters
    default_cv_results = lgb.cv(default_hyp, train_set, nfold = 5, num_boost_round = 10000, early_stopping_rounds = 100, 
                                metrics = 'auc', seed = RSEED)
    
    default_auc = default_cv_results['auc-mean'][-1]
    default_auc_std = default_cv_results['auc-stdv'][-1]
    
    # Locate the optimal hyperparameters
    hyp_results = hyp_results.sort_values('score', ascending = False).reset_index(drop = True)
    best_hyp = ast.literal_eval(hyp_results.loc[0, 'params'])
    best_random_score = hyp_results.loc[0, 'score']

    del best_hyp['n_estimators']

    # Cross validation with best hyperparameter values
    opt_cv_results = lgb.cv(best_hyp, train_set, nfold = 5, num_boost_round = 10000, early_stopping_rounds = 100, 
                            metrics = 'auc', seed = RSEED)

    opt_auc = opt_cv_results['auc-mean'][-1]
    opt_auc_std = opt_cv_results['auc-stdv'][-1]
    
    # Insert results into dataframe
    results.loc[0, 'default_auc'] = default_auc
    results.loc[0, 'default_auc_std'] = default_auc_std
    results.loc[0, 'random_search_auc'] = best_random_score
    results.loc[0, 'opt_auc'] = opt_auc
    results.loc[0, 'opt_auc_std'] = opt_auc_std
    
    # Extract the optimum number of estimators
    opt_n_estimators = len(opt_cv_results['auc-mean'])
    model = lgb.LGBMClassifier(n_estimators = opt_n_estimators, **best_hyp)
    
    # Fit on whole training set
    model.fit(train, train_labels)

    # Make predictions on testing data
    preds = model.predict_proba(test)[:, 1]

    # Make submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 
                               'TARGET': preds})

    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype(np.int32)
    
    # Make feature importances dataframe
    feature_importances = pd.DataFrame({'feature': feature_names,
                                        'importance': model.feature_importances_})

    return results, feature_importances, submission
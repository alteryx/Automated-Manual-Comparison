import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer

def plot_feature_importances(df, n = 15, color = 'blue', threshold = None):
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
    
    plt.style.use('fivethirtyeight')
    # Bar plot of n most important features
    df.loc[:(n - 1), :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = color, 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)
    plt.yticks(size = 14)
    plt.xticks(size = 16)
    plt.xlabel('Normalized Importance', size = 20); plt.ylabel(''); 
    plt.title(f'{n} Most Important Features', size = 24)
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

def feature_selection(feature_matrix, missing_threshold=90, correlation_threshold=0.95):
    """Feature selection for a dataframe."""
    
    feature_matrix = pd.get_dummies(feature_matrix)
    n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)

    # Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing['percent'] = 100 * (missing[0] / feature_matrix.shape[0])
    missing.sort_values('percent', ascending = False, inplace = True)

    # Missing above threshold
    missing_cols = list(missing[missing['percent'] > missing_threshold].index)
    n_missing_cols = len(missing_cols)

    # Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
    print('{} missing columns with threshold: {}.'.format(n_missing_cols,
                                                                        missing_threshold))
    
    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print('{} zero variance columns.'.format(n_zero_variance_cols))
    
    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop)
    
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                          correlation_threshold))
    
    total_removed = n_missing_cols + n_zero_variance_cols + n_collinear
    
    print('Total columns removed: ', total_removed)
    print('Shape after feature selection: {}.'.format(feature_matrix.shape))
    return feature_matrix

def mape(y_true, pred):
    """Metric for evaluating predictions"""
    mape = 100 * np.mean(abs(y_true - pred) / y_true)
    
    return mape

def evaluate(train, train_labels, test, test_labels):
    """Evaluate a training dataset with a standard sklearn model"""
    
    # Use the same model for each training set for now
    model = RandomForestRegressor(n_estimators = 100, 
                                  random_state = 50, n_jobs = -1)
    
    train = train.replace({np.inf: np.nan})
    test = test.replace({np.inf: np.nan})
    
    feature_names = list(train.columns)
    
    # Impute the missing values
    imputer = Imputer(strategy = 'median', axis = 1)
    train = imputer.fit_transform(train)
    test = imputer.transform(test)
    
    # Metric     
    mape_scorer = make_scorer(mape, greater_is_better = False)
    
    # Fit on the training data and make predictions
    model.fit(train, train_labels)
    preds = model.predict(test)
    
    cv_score = -1 * cross_val_score(model, train, train_labels, 
                                    scoring = mape_scorer, cv = 5)
    
    # Calculate the performance
    mape_score = mape(test_labels, preds)
    print('5-fold CV MAPE: {:.2f} with std: {:.2f}'.format(cv_score.mean(),cv_score.std()))
    print('Test MAPE: {:.2f}.'.format(mape_score))
    feature_importances = pd.DataFrame({'feature': feature_names, 
                                        'importance': model.feature_importances_})
    
    return preds, feature_importances

def random_search(train, train_labels):
    """Random Search on a set of features using a random forest"""
    
    # Metric     
    mape_scorer = make_scorer(mape, greater_is_better = False)
    scorer = mape_scorer

    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(50, 1000, num = 100)],
        'max_depth': [None] + [int(x) for x in np.linspace(4, 20)],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['auto', 'sqrt', 0.5, 'log2', None]
    }
    
    # Impute the missing values
    imputer = Imputer(strategy = 'median', axis = 1)
    train = imputer.fit_transform(train.replace({np.inf:np.nan}))
    
    # Make a model 
    reg = RandomForestRegressor(n_jobs = -1, random_state = 50)

    # RandomizedSearchCV object
    random = RandomizedSearchCV(reg, param_grid, n_iter = 100, scoring = scorer, 
                                verbose = 1, n_jobs = -1, cv = 5, random_state = 50)

    # Fit on the training data
    random.fit(train, train_labels)
    
    best_score = -1 * random.best_score_
    best_score_std = random.cv_results_['std_test_score'][np.argmax(random.cv_results_['mean_test_score'])]
    best_model = random.best_estimator_
    
    print('Best 5-fold CV Score: {:.2f} with std: {:.2f}.'.format(best_score, best_score_std))
    print('\nBest Hyperparameters:')
    print(random.best_params_)
    
    return best_model
# Data manipulation
import pandas as pd
import numpy as np

# modeling
import lightgbm as lgb

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

RSEED = 50

def format_data(features, free_memory = False):
    """Format a set of training and testing features joined together
       into separate sets for machine learning"""
    
    train = features[features['TARGET'].notnull()].copy()
    test = features[features['TARGET'].isnull()].copy()
    
    if free_memory:
        import gc
        gc.enable()
        del features
        gc.collect()
    
    train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))
    test_ids = list(test['SK_ID_CURR'])
    
    train = train.drop(columns = ['TARGET', 'SK_ID_CURR'])
    test = test.drop(columns = ['TARGET', 'SK_ID_CURR'])
    
    feature_names = list(train.columns)
    
    return train, train_labels, test, test_ids, feature_names


def cross_validate_and_submit(features, best_hyp, free_memory = False):
    """Function for assessing cross validation, calculating feature importances,
    and making a submission dataframe.
    
    Args:
        features (dataframe): dataset
        
    Returns:
        cv_results (dict): cross validation results. Keys will be 'auc-mean' and 'auc-stdv'.
        
        fi (dataframe): feature importance dataframe. Columns are 'feature' and 'importance'
        
        submission (dataframe): dataframe that can be submitted to the competition.
    
    """
    train, train_labels, test, test_ids, feature_names = format_data(features, free_memory)
    
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
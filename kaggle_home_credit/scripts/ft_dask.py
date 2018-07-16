# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

import featuretools.variable_types as vtypes

import sys
import psutil

import os

featurenames = ft.load_features('../input/features.txt')
print(len(featurenames))

# Read in the datasets and replace the anomalous values
app_train = pd.read_csv('../input/application_train.csv').replace({365243: np.nan})
app_test = pd.read_csv('../input/application_test.csv').replace({365243: np.nan})
bureau = pd.read_csv('../input/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('../input/bureau_balance.csv').replace({365243: np.nan})
cash = pd.read_csv('../input/POS_CASH_balance.csv').replace({365243: np.nan})
credit = pd.read_csv('../input/credit_card_balance.csv').replace({365243: np.nan})
previous = pd.read_csv('../input/previous_application.csv').replace({365243: np.nan})
installments = pd.read_csv('../input/installments_payments.csv').replace({365243: np.nan})

app_test['TARGET'] = np.nan

# Join together training and testing
app = app_train.append(app_test, ignore_index = True, sort = True)

# All ids should be integers
for index in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
    for dataset in [app, bureau, bureau_balance, cash, credit, previous, installments]:
        if index in list(dataset.columns):
            # Convert to integers after filling in missing values (not sure why values are missing)
            dataset[index] = dataset[index].fillna(0).astype(np.int64)

# Need `SK_ID_CURR` in every dataset
bureau_balance = bureau_balance.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']],
                                      on = 'SK_ID_BUREAU', how = 'left')


# Set the index for locating
for dataset in [app, bureau, bureau_balance, cash, credit, previous, installments]:
    dataset.set_index('SK_ID_CURR', inplace = True)

def create_partition(user_list, partition = None):
    """Creates a set of data with only the users in `user_list`. If `partition` is specified, then data is saved."""

    # Subset based on user list
    app_subset = app[app.index.isin(user_list)].copy().reset_index()
    bureau_subset = bureau[bureau.index.isin(user_list)].copy().reset_index()

    # Drop SK_ID_CURR from bureau_balance, cash, credit, and installments
    bureau_balance_subset = bureau_balance[bureau_balance.index.isin(user_list)].copy().reset_index(drop = True)
    cash_subset = cash[cash.index.isin(user_list)].copy().reset_index(drop = True)
    credit_subset = credit[credit.index.isin(user_list)].copy().reset_index(drop = True)
    previous_subset = previous[previous.index.isin(user_list)].copy().reset_index()
    installments_subset = installments[installments.index.isin(user_list)].copy().reset_index(drop = True)

    # Save files if partition is specified
    if partition:

        directory = '../../data/kaggle_home_credit/partitions/p%d' % (partition + 1)
        os.makedirs(directory)

        app_subset.to_csv('%s/app.csv' % directory, index = False)
        bureau_subset.to_csv('%s/bureau.csv' % directory, index = False)
        bureau_balance_subset.to_csv('%s/bureau_balance.csv' % directory, index = False)
        cash_subset.to_csv('%s/cash.csv' % directory, index = False)
        credit_subset.to_csv('%s/credit.csv' % directory, index = False)
        previous_subset.to_csv('%s/previous.csv' % directory, index = False)
        installments_subset.to_csv('%s/installments.csv' % directory, index = False)

        print('Saved all files in partition {} to {}.'.format(partition + 1,directory))

    return {'app': app_subset, 'bureau': bureau_subset, 'bureau_balance': bureau_balance_subset,
            'previous': previous_subset, 'installments': installments_subset,
            'cash': cash_subset, 'credit': credit_subset}

n = 20

# Break into n chunks
size = app.shape[0] // n

# Construct an id list
id_list = [list(app.iloc[i:i+size].index) for i in range(0, app.shape[0], size)]

# Test partition function
data_dict = create_partition(id_list[0])
print(data_dict.keys())

from itertools import chain

# Sanity check that we have not missed any ids
print('Number of ids in id_list:         {}.'.format(len(list(chain(*id_list)))))
print('Total length of application data: {}.'.format(len(app)))

app_types = {'FLAG_CONT_MOBILE': vtypes.Boolean, 'FLAG_DOCUMENT_10': vtypes.Boolean, 'FLAG_DOCUMENT_11': vtypes.Boolean, 'FLAG_DOCUMENT_12': vtypes.Boolean, 'FLAG_DOCUMENT_13': vtypes.Boolean, 'FLAG_DOCUMENT_14': vtypes.Boolean, 'FLAG_DOCUMENT_15': vtypes.Boolean, 'FLAG_DOCUMENT_16': vtypes.Boolean, 'FLAG_DOCUMENT_17': vtypes.Boolean, 'FLAG_DOCUMENT_18': vtypes.Boolean, 'FLAG_DOCUMENT_19': vtypes.Boolean, 'FLAG_DOCUMENT_2': vtypes.Boolean, 'FLAG_DOCUMENT_20': vtypes.Boolean, 'FLAG_DOCUMENT_21': vtypes.Boolean, 'FLAG_DOCUMENT_3': vtypes.Boolean, 'FLAG_DOCUMENT_4': vtypes.Boolean, 'FLAG_DOCUMENT_5': vtypes.Boolean, 'FLAG_DOCUMENT_6': vtypes.Boolean, 'FLAG_DOCUMENT_7': vtypes.Boolean, 'FLAG_DOCUMENT_8': vtypes.Boolean, 'FLAG_DOCUMENT_9': vtypes.Boolean, 'FLAG_EMAIL': vtypes.Boolean, 'FLAG_EMP_PHONE': vtypes.Boolean, 'FLAG_MOBIL': vtypes.Boolean, 'FLAG_PHONE': vtypes.Boolean, 'FLAG_WORK_PHONE': vtypes.Boolean, 'LIVE_CITY_NOT_WORK_CITY': vtypes.Boolean, 'LIVE_REGION_NOT_WORK_REGION': vtypes.Boolean, 'REG_CITY_NOT_LIVE_CITY': vtypes.Boolean, 'REG_CITY_NOT_WORK_CITY': vtypes.Boolean, 'REG_REGION_NOT_LIVE_REGION': vtypes.Boolean, 'REG_REGION_NOT_WORK_REGION': vtypes.Boolean, 'REGION_RATING_CLIENT': vtypes.Ordinal, 'REGION_RATING_CLIENT_W_CITY': vtypes.Ordinal, 'HOUR_APPR_PROCESS_START': vtypes.Ordinal}
previous_types = {'NFLAG_LAST_APPL_IN_DAY': vtypes.Boolean,
             'NFLAG_INSURED_ON_APPROVAL': vtypes.Boolean}

def entityset_from_partition(data_dict, return_featurenames = False):
    """Create an EntitySet from a partition of data in a dictionary"""

    # Extract the dataframes
    app = data_dict['app']
    bureau = data_dict['bureau']
    bureau_balance = data_dict['bureau_balance']
    previous = data_dict['previous']
    credit = data_dict['credit']
    installments = data_dict['installments']
    cash = data_dict['cash']

    # Add domain features to base dataframe
    app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
    app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
    app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
    app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
    app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis = 1)


    # Empty entityset
    es = ft.EntitySet(id = 'clients')

    # Entities with a unique index
    es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR',
                                  variable_types = app_types)

    es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')

    es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV',
                                  variable_types = previous_types)

    # Entities that do not have a unique index
    es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance,
                                  make_index = True, index = 'bureaubalance_index')

    es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash,
                                  make_index = True, index = 'cash_index')

    es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                                  make_index = True, index = 'installments_index')

    es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                                  make_index = True, index = 'credit_index')

    # Relationship between app_train and bureau
    r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

    # Relationship between bureau and bureau balance
    r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

    # Relationship between current app and previous apps
    r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

    # Relationships between previous apps and cash, installments, and credit
    r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
    r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
    r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

    # Add in the defined relationships
    es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                               r_previous_cash, r_previous_installments, r_previous_credit])

    # Domain Features from bureau
    es['bureau']['CREDIT_ACTIVE'].interesting_values = ['Active', 'Closed']


    credit_overdue = ft.Feature(es['bureau']['CREDIT_DAY_OVERDUE']) > 0.0
    credit_overdue = credit_overdue.rename('CREDIT_OVERDUE')

    credit_loan_rate = ft.Feature(es['bureau']['AMT_ANNUITY']) / ft.Feature(es['bureau']['AMT_CREDIT_SUM'])
    credit_loan_rate = credit_loan_rate.rename('PREVIOUS_OTHER_LOAN_RATE')


    # Domain Features from bureau balance
    balance_past_due = ft.Feature(es['bureau_balance']['STATUS']).isin(['1', '2', '3', '4', '5'])
    balance_past_due = balance_past_due.rename('PREVIOUS_OTHER_MONTHLY_PAST_DUE')


    # Domain Features from previous
    es['previous']['NAME_CONTRACT_STATUS'].interesting_values = ['Approved', 'Refused']

    previous_difference = ft.Feature(es['previous']['AMT_APPLICATION']) - ft.Feature(es['previous']['AMT_CREDIT'])
    previous_difference = previous_difference.rename('PREVIOUS_APPLICATION_RECEIVED_DIFFERENCE')

    previous_loan_rate = ft.Feature(es['previous']['AMT_ANNUITY']) / ft.Feature(es['previous']['AMT_CREDIT'])
    previous_loan_rate = previous_loan_rate.rename('PREVIOUS_LOAN_RATE')


    # Domain Features from credit
    es['credit']['NAME_CONTRACT_STATUS'].interesting_values = ['Active', 'Completed']

    credit_card_past_due = ft.Feature(es['credit']['SK_DPD']) > 0.0
    credit_card_past_due = credit_card_past_due.rename('CREDIT_CARD_PAST_DUE')


    # Domain Features from cash
    es['cash']['NAME_CONTRACT_STATUS'].interesting_values = ['Active', 'Completed']

    cash_past_due = ft.Feature(es['cash']['SK_DPD']) > 0.0
    cash_past_due = cash_past_due.rename('CASH_PAST_DUE')

    # Seed Features from installments
    installments_late = ft.Feature(es['installments']['DAYS_ENTRY_PAYMENT']) > ft.Feature(es['installments']['DAYS_INSTALMENT'])
    installments_late = installments_late.rename('INSTALLMENT_LATE')

    installments_low_payment = ft.Feature(es['installments']['AMT_PAYMENT']) < ft.Feature(es['installments']['AMT_INSTALMENT'])
    installments_low_payment = installments_low_payment.rename('INSTALLMENT_LOW')

    if return_featurenames:
        # List of seed features
        seed_features = [installments_low_payment, installments_late,
                               cash_past_due, credit_card_past_due,
                               previous_difference, previous_loan_rate,
                               balance_past_due, credit_loan_rate, credit_overdue]


        # Specify primitives
        agg_primitives =  ["sum", "max", "min", "mean", "count", "percent_true", "num_unique", "mode"]
        trans_primitives = ['percentile', 'and']
        where_primitives = ['percent_true', 'mean', 'sum']

        # Features only
        feature_names = ft.dfs(entityset=es, target_entity='app',
                               agg_primitives = agg_primitives,
                               trans_primitives = trans_primitives,
                               seed_features = seed_features,
                               where_primitives = where_primitives,
                               n_jobs = 1, verbose = 1, features_only = True,
                               max_depth = 2)

        return feature_names

    return es

# Test entityset function
es1 = entityset_from_partition(data_dict)
print(es1)

def feature_matrix_from_entityset(es, feature_names):

    """Run deep feature synthesis from an entityset and feature names"""

    feature_matrix = ft.calculate_feature_matrix(feature_names,
                                                 entityset=es)

    return feature_matrix


# # Test featurematrix function
# fm1 = feature_matrix_from_entityset(es1, featurenames)
# print(fm1.shape)

from dask import delayed
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler

from timeit import default_timer as timer

start = timer()

fms = []

for ids in id_list:
    ds = delayed(create_partition)(ids)
    es = delayed(entityset_from_partition)(ds)
    fm = delayed(feature_matrix_from_entityset)(es, feature_names = featurenames)
    fms.append(fm)

X = delayed(pd.concat)(fms, axis = 0)

with ProgressBar():
    feature_matrix = X.compute()

end = timer()
print('Elapsed: {:.5f}.'.format(end - start))

print(feature_matrix.shape)
feature_matrix.to_csv('feature_matrix.csv', chunksize = 1000)
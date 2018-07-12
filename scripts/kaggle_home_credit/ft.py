# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

# Read in the datasets
app_train = pd.read_csv('../input/application_train.csv').replace({365243: np.nan})
app_test = pd.read_csv('../input/application_test.csv').replace({365243: np.nan})
bureau = pd.read_csv('../input/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('../input/bureau_balance.csv').replace({365243: np.nan})
cash = pd.read_csv('../input/POS_CASH_balance.csv').replace({365243: np.nan})
credit = pd.read_csv('../input/credit_card_balance.csv').replace({365243: np.nan})
previous = pd.read_csv('../input/previous_application.csv').replace({365243: np.nan})
installments = pd.read_csv('../input/installments_payments.csv').replace({365243: np.nan})

# Add identifying column
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

# Append the dataframes
app = app_train.append(app_test, ignore_index = True, sort = True)

app['SK_ID_CURR'] = app['SK_ID_CURR'].fillna(0).astype(np.int64)

bureau['SK_ID_CURR'] = bureau['SK_ID_CURR'].fillna(0).astype(np.int64)
bureau['SK_ID_BUREAU'] = bureau['SK_ID_BUREAU'].fillna(0).astype(np.int64)

bureau_balance['SK_ID_BUREAU'] = bureau_balance['SK_ID_BUREAU'].fillna(0).astype(np.int64)

cash['SK_ID_PREV'] = cash['SK_ID_PREV'].fillna(0).astype(np.int64)
credit['SK_ID_PREV'] = credit['SK_ID_PREV'].fillna(0).astype(np.int64)
installments['SK_ID_PREV'] = installments['SK_ID_PREV'].fillna(0).astype(np.int64)

previous['SK_ID_CURR'] = previous['SK_ID_CURR'].fillna(0).astype(np.int64)
previous['SK_ID_PREV'] = previous['SK_ID_PREV'].fillna(0).astype(np.int64)


# Entity set with id applications
es = ft.EntitySet(id = 'clients')

import featuretools.variable_types as vtypes

app_types = {}

# Handle the Boolean variables:
for col in app:
    if app[col].nunique() == 2:
        app_types[col] = vtypes.Boolean

print('There are {} Boolean variables.'.format(len(app_types)))

app_types['REGION_RATING_CLIENT'] = vtypes.Ordinal
app_types['REGION_RATING_CLIENT_W_CITY'] = vtypes.Ordinal
app_types['WEEKDAY_APPR_PROCESS_START'] = vtypes.Ordinal
app_types['HOUR_APPR_PROCESS_START'] = vtypes.Ordinal


previous_types = {}

# Handle the Boolean variables:
for col in previous:
    if previous[col].nunique() == 2:
        previous_types[col] = vtypes.Boolean

print('There are {} Boolean variables.'.format(len(previous_types)))

installments = installments.drop(columns = ['SK_ID_CURR'])
credit = credit.drop(columns = ['SK_ID_CURR'])
cash = cash.drop(columns = ['SK_ID_CURR'])

app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']

app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis = 1)

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

import gc
gc.enable()
del app, app_test, app_train, bureau, bureau_balance, previous, cash, credit, installments
gc.collect()

# Relationship between app and bureau
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
# Print out the EntitySet
es

credit_active = ft.Feature(es['bureau']['CREDIT_ACTIVE']) != 'Closed'
credit_overdue = ft.Feature(es['bureau']['CREDIT_DAY_OVERDUE']) > 0.0

balance_past_due = ft.Feature(es['bureau_balance']['STATUS']).isin(['1', '2', '3', '4', '5'])

application_not_approved = ft.Feature(es['previous']['NAME_CONTRACT_STATUS']) != 'Approved'

credit_card_past_due = ft.Feature(es['credit']['SK_DPD']) > 0.0
credit_card_active = ft.Feature(es['credit']['NAME_CONTRACT_STATUS']) == 'Active'

cash_active = ft.Feature(es['cash']['NAME_CONTRACT_STATUS']) == 'Active'
cash_past_due = ft.Feature(es['cash']['SK_DPD']) > 0.0

installments_late = ft.Feature(es['installments']['DAYS_ENTRY_PAYMENT']) > ft.Feature(es['installments']['DAYS_INSTALMENT'])
installments_low_payment = ft.Feature(es['installments']['AMT_PAYMENT']) < ft.Feature(es['installments']['AMT_INSTALMENT'])

seed_features = [installments_low_payment, installments_late,
                 cash_past_due, cash_active,
                 credit_card_active, credit_card_past_due,
                 application_not_approved, balance_past_due,
                 credit_overdue, credit_active]

agg_primitives =  ["sum", "max", "min", "mean", "count", "percent_true", "num_unique", "mode"]
trans_primitives = ['percentile', 'and']

feature_names = ft.dfs(entityset=es, target_entity='app',
                       agg_primitives = agg_primitives,
                       trans_primitives = trans_primitives,
                       seed_features = seed_features,
                       n_jobs = 4, verbose = 1, features_only = True,
                       max_depth = 2)

import sys

print('Size of entityset: {} gb.'.format(sys.getsizeof(es) / 1e9))

import psutil

print('Number of cpus detected: {}.'.format(psutil.cpu_count()))
print('Total virtual memory detected: {} gb.'.format(psutil.virtual_memory().total / 1e9))

feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='app',
                                       agg_primitives = agg_primitives,
                                       trans_primitives = trans_primitives,
                                       seed_features = seed_features,
                                       n_jobs = 1, verbose = 1, features_only = False,
                                       max_depth = 2, chunk_size = 100)

feature_matrix.reset_index(inplace = True)
feature_matrix.to_csv('../input/feature_matrix_ft.csv', index = False, chunksize = 100)
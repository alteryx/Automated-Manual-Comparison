# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

import json

# featuretools for automated feature engineering
import featuretools as ft
import featuretools.variable_types as vtypes


featurenames = ft.load_features('../input/features.txt')
print('Number of features: {}'.format(len(featurenames)))

print('Reading in data')
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

for index in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
    for dataset in [app, bureau, bureau_balance, cash, credit, previous, installments]:
        if index in list(dataset.columns):
            dataset[index] = dataset[index].fillna(0).astype(np.int64)


# Entity set with id applications
es = ft.EntitySet(id = 'clients')

app_types = {}

# Handle the Boolean variables:
for col in app:
    if (app[col].nunique() == 2) and (app[col].dtype == float):
        app_types[col] = vtypes.Boolean

# Remove the `TARGET`
del app_types['TARGET']

# Ordinal variables
app_types['REGION_RATING_CLIENT'] = vtypes.Ordinal
app_types['REGION_RATING_CLIENT_W_CITY'] = vtypes.Ordinal
app_types['HOUR_APPR_PROCESS_START'] = vtypes.Ordinal

with open('../input/app_types.txt', 'w') as f:
    f.write(str(app_types))

previous_types = {}

# Handle the Boolean variables:
for col in previous:
    if (previous[col].nunique() == 2) and (previous[col].dtype == float):
        previous_types[col] = vtypes.Boolean

with open('../input/previous_types.txt', 'w') as f:
    f.write(str(previous_types))

# Drop the ids
installments = installments.drop(columns = ['SK_ID_CURR'])
credit = credit.drop(columns = ['SK_ID_CURR'])
cash = cash.drop(columns = ['SK_ID_CURR'])


# Add domain features to app
app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis = 1)

# Adding Entities
print('Adding entities')
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


print('EntitySet Complete')
# Print out the EntitySet
print(es)



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

# List of seed features
seed_features = [installments_low_payment, installments_late,
                       cash_past_due, credit_card_past_due,
                       previous_difference, previous_loan_rate,
                       balance_past_due, credit_loan_rate, credit_overdue]


# Specify primitives
agg_primitives =  ["sum", "max", "min", "mean", "count", "percent_true", "num_unique", "mode"]
trans_primitives = ['percentile', 'and']
where_primitives = ['percent_true', 'mean', 'sum']

# print('DFS for featurenames')

# # Deep feature synthesis with domain knowledge (only features)
# feature_names = ft.dfs(entityset=es, target_entity='app',
#                        agg_primitives = agg_primitives,
#                        trans_primitives = trans_primitives,
#                        seed_features = seed_features,
#                        where_primitives = where_primitives,
#                        n_jobs = -1, verbose = 1, features_only = True,
#                        max_depth = 2)

# ft.save_features(feature_names, '../input/features.txt')

import sys
print('Total size of entityset: {:.5f} gb.'.format(sys.getsizeof(es) / 1e9))


import psutil

print('Total number of cpus detected: {}.'.format(psutil.cpu_count()))
print('Total size of system memory: {:.5f} gb.'.format(psutil.virtual_memory().total / 1e9))

print('Running DFS')
feature_matrix, feature_names = ft.calculate_feature_matrix(featurenames,
                                                            entityset=es,
                                                            n_jobs = 1,
                                                            verbose = 1,
                                                            chunk_size = 100)


feature_matrix.reset_index(inplace = True)
feature_matrix.to_csv('../input/feature_matrix.csv', index = False)
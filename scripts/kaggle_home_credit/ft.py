# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft


# ### Read in Data



# Read in the datasets and replace the anomalous values
app_train = pd.read_csv('../input/application_train.csv').replace({365243: np.nan})
app_test = pd.read_csv('../input/application_test.csv').replace({365243: np.nan})
bureau = pd.read_csv('../input/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('../input/bureau_balance.csv').replace({365243: np.nan})
cash = pd.read_csv('../input/POS_CASH_balance.csv').replace({365243: np.nan})
credit = pd.read_csv('../input/credit_card_balance.csv').replace({365243: np.nan})
previous = pd.read_csv('../input/previous_application.csv').replace({365243: np.nan})
installments = pd.read_csv('../input/installments_payments.csv').replace({365243: np.nan})


# We will join together the training and testing datasets to make sure we build the same features for each set. Later, after the feature matrix is built, we can separate out the two sets.

# In[3]:


app_test['TARGET'] = np.nan

# Join together training and testing
app = app_train.append(app_test, ignore_index = True, sort = True)


# Several of the keys are an incorrect data type (floats) so we need to make these all the same (integers) for adding relationships.

# In[29]:


for index in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
    for dataset in [app, bureau, bureau_balance, cash, credit, previous, installments]:
        if index in list(dataset.columns):
            dataset[index] = dataset[index].fillna(0).astype(np.int64)




# Entity set with id applications
es = ft.EntitySet(id = 'clients')



import featuretools.variable_types as vtypes


# In[32]:


app_types = {}

# Handle the Boolean variables:
for col in app:
    if (app[col].nunique() == 2) and (app[col].dtype == float):
        app_types[col] = vtypes.Boolean

# Remove the `TARGET`
del app_types['TARGET']

print('There are {} Boolean variables in the application data.'.format(len(app_types)))


# In[33]:


# Ordinal variables
app_types['REGION_RATING_CLIENT'] = vtypes.Ordinal
app_types['REGION_RATING_CLIENT_W_CITY'] = vtypes.Ordinal
app_types['HOUR_APPR_PROCESS_START'] = vtypes.Ordinal


# The `previous` table is the only other `entity` that has features which should be recorded as Boolean. Correctly identifying the type of column will prevent featuretools from making irrelevant features such as the mean or max of a `Boolean`.

# In[34]:


previous_types = {}

# Handle the Boolean variables:
for col in previous:
    if (previous[col].nunique() == 2) and (previous[col].dtype == float):
        previous_types[col] = vtypes.Boolean

print('There are {} Boolean variables in the previous data.'.format(len(previous_types)))


# In addition to identifying Boolean variables, we want to make sure featuretools does not create nonsense features such as statistical aggregations (mean, max, etc.) of ids. The `credit`, `cash`, and `installments` data all have the `SK_ID_CURR` variable. However, we do not actually need this variable in these dataframes because we link them to `app` through the `previous` dataframe with the `SK_ID_PREV` variable. We don't want to make features from `SK_ID_CURR` since it is an arbitrary id and should have no predictive power. Features like the mean of the id are irrelevant and would only slow down model training and probably lead to poorer model performance.
#
# Our options to handle these variables is either to tell featuretools to ignore them, or to drop the features before including them in the entityset. We will take the latter approach.

# In[35]:


installments = installments.drop(columns = ['SK_ID_CURR'])
credit = credit.drop(columns = ['SK_ID_CURR'])
cash = cash.drop(columns = ['SK_ID_CURR'])


# ## Adding Entities
#
# Now we define each entity, or table of data, and add it to the `EntitySet`. We need to pass in an index if the table has one or `make_index = True` if not. In the cases where we need to make an index, we must supply a name for the index. We also need to pass in the dictionary of variable types if there are any specific variables we should identify. The following code adds all eight tables to the `EntitySet`.

# In[39]:


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
# Print out the EntitySet
print(es)


app_loan_rate = ft.Feature(es['app']['AMT_ANNUITY']) / ft.Feature(es['app']['AMT_CREDIT'])
app_loan_rate = app_loan_rate.rename('LOAN_RATE')

app_credit_income_ratio = ft.Feature(es['app']['AMT_CREDIT']) / ft.Feature(es['app']['AMT_INCOME_TOTAL'])
app_credit_income_ratio = app_credit_income_ratio.rename('CREDIT_INCOME_RATIO')

app_employed_birth_ratio = ft.Feature(es['app']['DAYS_EMPLOYED']) / ft.Feature(es['app']['DAYS_BIRTH'])
app_employed_birth_ratio = app_employed_birth_ratio.rename('EMPLOYED_BIRTH_RATIO')

app_ext_source_sum = ft.Feature(es['app']['EXT_SOURCE_1']) + ft.Feature(es['app']['EXT_SOURCE_2']) + ft.Feature(es['app']['EXT_SOURCE_3'])
app_ext_source_sum = app_ext_source_sum.rename('EXT_SOURCE_SUM')

app_ext_source_mean = (ft.Feature(es['app']['EXT_SOURCE_1']) + ft.Feature(es['app']['EXT_SOURCE_2']) + ft.Feature(es['app']['EXT_SOURCE_3'])) / 3
app_ext_source_mean = app_ext_source_mean.rename('EXT_SOURCE_MEAN')


# #### Domain Features from bureau



es['bureau']['CREDIT_ACTIVE'].interesting_values = ['Active', 'Closed']


credit_overdue = ft.Feature(es['bureau']['CREDIT_DAY_OVERDUE']) > 0.0
credit_overdue = credit_overdue.rename('CREDIT_OVERDUE')

credit_loan_rate = ft.Feature(es['bureau']['AMT_ANNUITY']) / ft.Feature(es['bureau']['AMT_CREDIT_SUM'])
credit_loan_rate = credit_loan_rate.rename('PREVIOUS_OTHER_LOAN_RATE')


# #### Domain Features from bureau balance



balance_past_due = ft.Feature(es['bureau_balance']['STATUS']).isin(['1', '2', '3', '4', '5'])
balance_past_due = balance_past_due.rename('PREVIOUS_OTHER_MONTHLY_PAST_DUE')


# #### Domain Features from previous


es['previous']['NAME_CONTRACT_STATUS'].interesting_values = ['Approved', 'Refused']

previous_difference = ft.Feature(es['previous']['AMT_APPLICATION']) - ft.Feature(es['previous']['AMT_CREDIT'])
previous_difference = previous_difference.rename('PREVIOUS_APPLICATION_RECEIVED_DIFFERENCE')

previous_loan_rate = ft.Feature(es['previous']['AMT_ANNUITY']) / ft.Feature(es['previous']['AMT_CREDIT'])
previous_loan_rate = previous_loan_rate.rename('PREVIOUS_LOAN_RATE')


# #### Domain Features from credit



es['credit']['NAME_CONTRACT_STATUS'].interesting_values = ['Active', 'Completed']


credit_card_past_due = ft.Feature(es['credit']['SK_DPD']) > 0.0
credit_card_past_due = credit_card_past_due.rename('CREDIT_CARD_PAST_DUE')


# #### Domain Features from cash


es['cash']['NAME_CONTRACT_STATUS'].interesting_values = ['Active', 'Completed']



cash_past_due = ft.Feature(es['cash']['SK_DPD']) > 0.0
cash_past_due = cash_past_due.rename('CASH_PAST_DUE')


# #### Seed Features from installments



installments_late = ft.Feature(es['installments']['DAYS_ENTRY_PAYMENT']) > ft.Feature(es['installments']['DAYS_INSTALMENT'])
installments_late = installments_late.rename('INSTALLMENT_LATE')

installments_low_payment = ft.Feature(es['installments']['AMT_PAYMENT']) < ft.Feature(es['installments']['AMT_INSTALMENT'])
installments_low_payment = installments_low_payment.rename('INSTALLMENT_LOW')


seed_features = [installments_low_payment, installments_late,
                       cash_past_due, credit_card_past_due,
                       previous_difference, previous_loan_rate,
                       balance_past_due, credit_loan_rate, credit_overdue,
                       app_credit_income_ratio, app_employed_birth_ratio,
                       app_ext_source_mean, app_ext_source_sum, app_loan_rate]


# ## Using DFS to build on Domain Knowledge



# Specify primitives
agg_primitives =  ["sum", "max", "min", "mean", "count", "percent_true", "num_unique", "mode"]
trans_primitives = ['percentile', 'and']
where_primitives = ['percent_true', 'mean', 'sum']


# Deep feature synthesis with domain knowledge (only features)
feature_names = ft.dfs(entityset=es, target_entity='app',
                       agg_primitives = agg_primitives,
                       trans_primitives = trans_primitives,
                       seed_features = seed_features,
                       where_primitives = where_primitives,
                       n_jobs = -1, verbose = 1, features_only = True,
                       max_depth = 2)


# ## Run Full Deep Feature Synthesis

import sys
print('Total size of entityset: {:.5f} gb.'.format(sys.getsizeof(es) / 1e9))


import psutil

print('Total number of cpus detected: {}.'.format(psutil.cpu_count()))
print('Total size of system memory: {:.5f} gb.'.format(psutil.virtual_memory().total / 1e9))



feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='app',
                                       agg_primitives = agg_primitives,
                                       trans_primitives = trans_primitives,
                                       seed_features = seed_features,
                                       where_primitives = where_primitives,
                                       n_jobs = 1, verbose = 1, features_only = False,
                                       max_depth = 2, chunk_size = 1000)



feature_matrix.reset_index(inplace = True)
feature_matrix.to_csv('../input/feature_matrix.csv', index = False)
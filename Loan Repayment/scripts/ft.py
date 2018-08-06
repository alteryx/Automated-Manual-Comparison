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

def convert_types(df):
    """Convert data types in a pandas dataframe. Purpose is to reduce size of dataframe."""
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    return df

print(f"""Total memory before converting types: \
{round(np.sum([x.memory_usage().sum() / 1e9 for x in 
[app, bureau, bureau_balance, cash, credit, previous, installments]]), 2)} gb.""")

# Convert types to reduce memory usage
app = convert_types(app)
bureau = convert_types(bureau)
bureau_balance = convert_types(bureau_balance)
cash = convert_types(cash)
credit = convert_types(credit)
previous = convert_types(previous)
installments = convert_types(installments)

print(f"""Total memory after converting types: \
{round(np.sum([x.memory_usage().sum() / 1e9 for x in 
[app, bureau, bureau_balance, cash, credit, previous, installments]]), 2)} gb.""")

# Entity set with id applications
es = ft.EntitySet(id = 'clients')

# Adding Entities
print('Adding entities')

# Entities with a unique index
es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR')

es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')

es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV')

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


# Specify primitives
agg_primitives =  ["sum", "max", "min", "mean", "count", "percent_true", "num_unique", "mode"]
trans_primitives = ['percentile', 'and']


# Run if we want to just generate the features
# feature_names = ft.dfs(entityset=es, target_entity='app',
#                                        agg_primitives = agg_primitives,
#                                        trans_primitives = trans_primitives
#                                        n_jobs = 1, verbose = 1, 
#                                        features_only = True,
#                                        max_depth = 2)

# ft.save_features(feature_names, '../input/features.txt')

import sys
import psutil

print('Total size of entityset: {:.5f} gb.'.format(sys.getsizeof(es) / 1e9))
print('Total number of cpus detected: {}.'.format(psutil.cpu_count()))
print('Total size of system memory: {:.5f} gb.'.format(psutil.virtual_memory().total / 1e9))

print('Deep Feature Synthesis')
feature_matrix = ft.calculate_feature_matrix(featurenames,
                                             entityset=es,
                                             n_jobs = 1,
                                             verbose = 1,
                                             chunk_size = 100)


feature_matrix.reset_index(inplace = True)
feature_matrix.to_csv('../input/feature_matrix.csv', index = False)
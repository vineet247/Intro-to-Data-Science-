# Add all relevant imports here
import os
import json
import numpy as np
import pandas as pd

import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import sklearn.naive_bayes as naive_bayes
import sklearn.linear_model as linear_model

from sklearn import impute
from sklearn import preprocessing as preproc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error, roc_curve, auc

from google.colab import files
from google.colab import drive
# uploaded = files.upload()

# Accessing Google sheets
#!pip install --upgrade -q gspread
from google.colab import auth
auth.authenticate_user()
import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())
worksheet = gc.open('AutoKaggle').worksheet('Metadata')
# get_all_values gives a list of rows
_rows = worksheet.get_all_values()
# Convert to a DataFrame and render.
import pandas as pd
rows = pd.DataFrame.from_records(_rows)
new_header = rows.iloc[0] #grab the first row for the header
rows = rows[1:] #take the data less the header row
rows.columns = new_header #set the header row as the df header



def alpha_to_number(alpha_key):
  return sum([(ord(alpha)-64)*(26**ind) for ind, alpha in enumerate(list(alpha_key)[::-1])]) - 1


def parseMetaData(row_id):

  # Parse data from MetaData for each row
 column_key = {'name': 'C', 'columns': 'W', 'estimator_func_call': 'AU', 'target_name': 'AC', 'output_type': 'AA', 'performance_metric': 'BB', 'feature_selector': 'AL'}
 column_key = dict(map(lambda kv: (kv[0], alpha_to_number(kv[1])), column_key.items()))

 metadata['competition_name'] = rows.loc[row_id][column_key['name']]
 metadata['estimator'] = rows.loc[row_id][column_key['estimator_func_call']]
 metadata['target_column'] = rows.loc[row_id][column_key['target_name']]
 metadata['output_type'] = rows.loc[row_id][column_key['output_type']].split(',')
 metadata['metric'] = rows.loc[row_id][column_key['performance_metric']]
 metadata['feature_selector'] = rows.loc[row_id][column_key['feature_selector']]
 columns = rows.loc[row_id][column_key['columns']]

 # Parse column information
 numeric_columns = []
 unwanted_columns = []
 categorical_columns = []
 columns_data = [x.strip() for x in columns[1:-1].split(';')]
 for ind, val in enumerate(columns_data):
  if ind%3 == 2:
   if (val == "numeric" or val == "integer" or val == "real"):
    numeric_columns.append(columns_data[ind-1])
   elif(val == "categorical"):
    categorical_columns.append(columns_data[ind-1])
   elif(val == "unwanted" or val == "string" or val == 'dateTime'):
    unwanted_columns.append(columns_data[ind-1])
   else:
    pass

  metadata['numeric_columns'] = numeric_columns
  metadata['unwanted_columns'] = unwanted_columns
  metadata['categorical_columns'] = categorical_columns

  # Remove target from features columns
  if metadata['target_column'] in metadata['numeric_columns']:
   metadata['numeric_columns'].remove(metadata['target_column'])
  if metadata['target_column'] in metadata['categorical_columns']:
   metadata['categorical_columns'].remove(metadata['target_column'])
  if metadata['target_column'] in metadata['unwanted_columns']:
   metadata['unwanted_columns'].remove(metadata['target_column'])

  print("competition: ",metadata['competition_name'])
  print("numeric columns: ",metadata['numeric_columns'])
  print("categorical columns: ",metadata['categorical_columns'])
  print("unwanted columns: ",metadata['unwanted_columns'])
  print("target column: ",metadata['target_column'])
  print("metric: ",metadata['metric'])
  print("feature selector: ",metadata['feature_selector'])
  print("estimator: ",metadata['estimator'])


  # Mount Google Drive
drive.mount('/content/gdrive')

def preprocessing(train_df):

  # drop unwanted columns
  if metadata['unwanted_columns']:
    train_df.drop(metadata['unwanted_columns'], axis=1, inplace=True)

  X = train_df.drop(metadata['target_column'], 1)
  y = train_df[metadata['target_column']]

  # treat missing values
  pd.set_option('mode.chained_assignment', None) # used to subside the panda's chain assignment warning
  imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  for col in metadata['numeric_columns']:
    X[[col]] = imp.fit_transform(X[[col]])

  # Categorial transform
  for col in metadata['categorical_columns']:
    col_dummies = pd.get_dummies(X[col], dummy_na=True)
    X = pd.concat([X, col_dummies], axis=1)
  X.drop(metadata['categorical_columns'], axis=1, inplace=True)

  # Feature normalization
  X[metadata['numeric_columns']] = preproc.scale(X[metadata['numeric_columns']])

  X_train, X_test, y_train, y_test = train_test_split(X, y)

  return X_train, X_test, y_train, y_test


 # Feature Extraction
def feature_extraction():
  pass


 # Feature Selection
def feature_selection(X_train, X_test, y_train, y_test):

  selector = eval(metadata['feature_selector'])
  X_train = selector.fit_transform(X_train, y_train)
  X_test = selector.fit_transform(X_test, y_test)
  return X_train, X_test, y_train, y_test


#Estimation
def estimation(X_train, X_test, y_train, y_test):

 model = eval(metadata['estimator'])

 model.fit(X_train, y_train)
 predict = model.predict(X_test)
 if metadata['metric'] == "rmse":
  error = np.sqrt(mean_squared_error(y_test, predict))
 elif metadata['metric'] == "accuracy":
  error = accuracy_score(y_test, predict)
 elif metadata['metric'] == "auc":
  fpr, tpr, _ = roc_curve(y_test, predict)
  error = auc(fpr, tpr)
 print(error)


 # Postprocessing
def postprocessing():
  pass


row_ids = [11]
metadata = {}

# Set your current working directory
cwd = "/gdrive/My Drive/Introduction to Data Science Spring 2019 Term Project/vv913/"

for row_id in row_ids:

  # Parsing MetaData updates the metadata dict
  metadata.clear()
  print("************************************************************")
  parseMetaData(row_id)
  competition_dir = cwd + metadata['competition_name'] + '/data/trainData.csv'

  train_df = pd.read_csv('/content/gdrive/My Drive/Introduction to Data Science Spring 2019 Term Project/vv913/talkingdata-adtracking-fraud-detection/data/trainData.csv')

  X_train, X_test, y_train, y_test = preprocessing(train_df)
  if metadata['feature_selector']:
     X_train, X_test, y_train, y_test = feature_selection(X_train, X_test, y_train, y_test)
  estimation(X_train, X_test, y_train, y_test)
  print("************************************************************")

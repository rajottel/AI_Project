import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from skimpy import skim
from sklearn.impute import KNNImputer
from numpy import isnan
import warnings
warnings.filterwarnings("ignore")
print_header = lambda msg: print(f"{msg}\n{'-'*len(msg)}")
# import datasets
train_data = pd.read_csv("data/aps_failure_training_modified.csv", na_values="na")
test_data = pd.read_csv("data/aps_failure_test_modified.csv", na_values="na")
# # impute starts below:
#
# separate train dataset into class and all other features, where class is the y and all other features are the x
y_train_data = train_data['class']
x_train_data = train_data.drop('class', axis=1)

# separate test dataset into class and all other features
y_test_data = test_data['class']
x_test_data = test_data.drop('class', axis=1)

train_columns = x_train_data.columns
test_columns = x_test_data.columns
# print(columns)
# x_train_data1 = x_train_data.rename(columns=x_train_dict)
print(x_train_data.head())
print_header("Dataset before imputation takes place")
print('Train Missing: ', x_train_data.isna().sum().sum())
print('Test Missing: ', x_test_data.isna().sum().sum())
# skim(x_train_data)
# print(y_train_data)
y_train_column = ["class"]
y_data = pd.DataFrame(y_train_data, columns=y_train_column)
y_data1 = pd.DataFrame(y_test_data, columns=y_train_column)
# train_data1 = y_data.join(x_train_data)
# print(train_data1.head())

# from https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/
imputer = KNNImputer()
imputer.fit(x_train_data)
x_train_data1 = imputer.transform(x_train_data)
x_test_data1 = imputer.transform(x_test_data)
print_header("Dataset after imputation takes place")
print('Train Missing: %d' % sum(isnan(x_train_data1).flatten()))
print('Test Missing: %d' % sum(isnan(x_test_data1).flatten()))
x_train_data2 = pd.DataFrame(x_train_data1, columns=train_columns)
x_test_data2 = pd.DataFrame(x_test_data1, columns=test_columns)
# print("X Train Data:\n", x_train_data2.head())
# print("X Test Data:\n", x_test_data2.head())


#skim(x_train_data1)

# concatenate y and x dataframes back to one dataframe in order to save csv
train_data1 = y_data.join(x_train_data2)
test_data1 = y_data1.join(x_test_data2)
print_header("Imputed Training Set Head")
print(train_data1.head())
print_header("Imputed Test Set Head")
print(test_data1.head())

# test_data1.to_csv('data/aps_failure_test_imputed.csv', index=False)
# train_data1.to_csv('data/aps_failure_training_imputed.csv', index=False)


import pandas as pd
import numpy as np
from skimpy import skim

train_data = pd.read_csv("data/aps_failure_training_set.csv", na_values="na")
print("Training Dataset Head:")
print(train_data.head())

test_data = pd.read_csv("data/aps_failure_test_set.csv", na_values="na")
print("\nTest Dataset Head:")
print(test_data.head())
skim(train_data)

# drop class feature
train_data1 = train_data.drop(columns='class')
# train dataset describe
train_info = train_data.describe()
# remove features with single values for all rows
singlevalue_features = []
for i in train_data1.columns:
    if train_info[i]['std'] == 0:
        singlevalue_features.append(i)
train_data1 = train_data.drop(columns=singlevalue_features)
skim(train_data1)
# here
print("Single value features dropped from dataset:", singlevalue_features)

# print("\nAppended Train Dataset shape:")
# print(train_data1.head())

# Creating a dictionary whose keys are the column names and values are the percentage of missing values
train_na_count = {k: list(train_data1.isna().sum()*100/train_data1.shape[0])[i] for i, k in enumerate(train_data1.columns)}
# Sorting the dictionary in descending order based on the percentage of missing values
train_na_count_sorted = {k: v for k, v in sorted(train_na_count.items(), key=lambda item: item[1], reverse=True)}
print("\nSorted NA count for training dataset by percentage of NA values:")
print(train_na_count_sorted)

# list features with missing values over 60%
# from https://www.geeksforgeeks.org/python-remove-keys-with-values-greater-than-k-including-mixed-values/
dropped_features = {}
for j in train_na_count_sorted:
    if train_na_count_sorted[j] > 60.0:
        dropped_features[j] = train_na_count_sorted[j]
print("\nThe features being removed are those with over 60% of data missing, which includes:", dropped_features.keys())

# remove features with greater than 60% missing data and remove rows with missing data in features with
# less than 10% missing data
# from https://www.plus2net.com/python/pandas-dataframe-dropna-thresh.php
na_10_percent = {k for k, v in train_na_count.items() if v < 10}
na_list = list(dropped_features.keys())
removed_features = na_list + singlevalue_features
print(removed_features)
print(na_list)
train_data2 = train_data1.drop(na_list, axis=1)
# here
# skim(train_data2)
train_data3 = train_data2.dropna(subset=na_10_percent)
# skim(train_data3)

print("\nShape of training dataset with rows that have NA entries in features with less than 10% missing data removed: ")
print(train_data3.shape)
print("head of new training dataset:")
print(train_data3.head())

# remove features from test dataset
test_data1 = test_data.drop(removed_features, axis=1)
print("Test Data prior to feature removal and row removal:", test_data.shape)
print("Test Data after feature removal and row removal:", test_data1.shape)
print("Training Data prior to feature removal and row removal:", train_data.shape)
print("Training Data after feature removal and row removal:", train_data3.shape)

# test_data1.to_csv('data/aps_failure_test_modified.csv', index=False)
# train_data3.to_csv('data/aps_failure_training_modified.csv', index=False)


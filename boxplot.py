from os import listdir
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import pyplot

test_data = pd.read_csv("Data/aps_failure_test_set.csv")
train_data = pd.read_csv("Data/aps_failure_training_set.csv")

# labelencoder = LabelEncoder()
# test_data['class'] = labelencoder.fit_transform(test_data['class'])
# train_data['class'] = labelencoder.fit_transform(train_data['class'])

def get_correct_label(y):
      return y.replace(['neg','pos'],[0,1])

print(train_data['class'].unique())
train_data['class'] = get_correct_label(train_data['class'])
print(train_data['class'].unique())

# Creating a dictionary whose keys are the column names and values are the percentage of missing values
nan_count = {k:list(train_data.isna().sum()*100/train_data.shape[0])[i] for i,k in enumerate(train_data.columns)}

# Sorting the dictionary in descending order based on the percentage of missing values
nan_count = {k: v for k, v in sorted(nan_count.items(), key=lambda item: item[1],reverse=True)}

# def remove_na(train_data, nan_feat):
#     """
#     This function removes features having more than 70%
#     missing data, and removes rows that have NA values
#     from features that have less than 5% missing data
#
#     """
#
#     # Removing features having more than 70% NA
#     train_data = train_data.dropna(axis=1, thresh=18000)
#
#     # Removing rows having NA from above created list of features
#     train_data = train_data.dropna(subset=nan_feat)
#
#     # Reset Index values
#     train_data = train_data.reset_index(drop=True)
#     return test_data
#
#
# print("Earlier shape of x:", train_data.shape)
#
# # List of features having less than 5% NA
# na_5 = [k for k, v in nan_count.items() if v < 5]
#
# x = remove_na(x, na_5)
# print("Shape after removal of rows and columns:", x.shape)
# print("Number of features having missing values below 5%:", len(na_5))

columns = train_data.columns
columns = list(columns)

plots = []
for i in train_data.columns:
    sns.boxplot(train_data[i])
    plt.show()

#test_data.boxplot('aa_000', 'ab_000', 'ac_000', 'ad_000', 'ae_000', 'af_000', 'ag_000')
plt.show()

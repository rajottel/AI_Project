# Initializing packages
import pandas as pd
import numpy as np
from skimpy import skim

# import training data
train_data = pd.read_csv("data/aps_failure_training_set.csv")

# replace 'na' entries from the dataset to np.nan so that the NaN values can be converted from string
train_data1 = train_data.replace('na', np.nan)

# removing the class feature to visualize just the feature data, as skimpy is not useful for pos/neg
train_data1 = train_data1.drop(columns='class')

# cols represents all columns with the object (string) datatype
cols = train_data1.columns[train_data1.dtypes.eq('object')]

# this is replacing all errors (non-numerical values)
# to NA values, then converting all the object columns
# in the above statement into numerical datatypes
# from https://stackoverflow.com/questions/36814100/pandas-to-numeric-for-multiple-columns
train_data1[cols] = train_data1[cols].apply(pd.to_numeric, errors='coerce')

# skim is a function that produces summary statistics of a given dataframe
# we are using skim to visualize data such as the NA count, % of the column that is NA,
# the mean, standard deviation, and histogram for each column, as well as the
# p0, p25, p75, and p100 percentiles.
skim(train_data1)

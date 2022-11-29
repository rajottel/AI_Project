import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# lines 8-10 from https://kearnz.github.io/autoimpute-tutorials/
import warnings
warnings.filterwarnings("ignore")
print_header = lambda msg: print(f"{msg}\n{'-'*len(msg)}")

# This part of the code gives a plot of each identifier and how many bins they have
og_dataframe = pd.read_csv("Data/aps_failure_training_set.csv")
dataframe = pd.read_csv("Data/aps_failure_training_imputed.csv")
test_data = pd.read_csv("Data/aps_failure_test_imputed.csv")
count = 0
# Reference from https://stackoverflow.com/questions/29947574/splitting-at-underscore-in-python-and-storing-the-first-value
first = dataframe.columns.str.split('_').str[0].tolist()
first.pop(0)
column_names = dataframe.columns.tolist()
column_names.pop(0)
list_of_indicators = sorted(set(first))
i = 0
count_first = []
for j in list_of_indicators:
    count_first.append(first.count(list_of_indicators[i]))
    i += 1
print_header("List of Indicators, and counts:")
print(list_of_indicators)
print(count_first)
plt.figure(figsize=(24, 5))
sb.barplot(x=list_of_indicators, y=count_first, color="green")
plt.show()

# This part of the program creates the iterator for Histogram with a value 10
u=0
BinsHist = []
for j in count_first:
    if count_first[u] == 10:
        BinsHist.append(u)
    u += 1

# This part of the program creates a list of the histogram identifier with 10 bins
HistIdenWith10Bins = []
for j in BinsHist:
    HistIdenWith10Bins.append(list_of_indicators[j])
print("These are the indicators with 10 bins each: ", HistIdenWith10Bins)

Hist70Features = []
index70F = []
g = 0
n = 0
for j in first:
    for g in HistIdenWith10Bins:
        if j == g:
            index70F.append(n)
    n += 1

for j in index70F:
    Hist70Features.append(column_names[j])
print("These are the the histogram and bins: ", Hist70Features)

# Get only x train data from the dataframe
imputed_x_data = dataframe.drop('class', axis=1)

# Get the y train from the unprocessed dataframe
train_set_y = dataframe['class']

# I am not sure about this one
imputed_x_histogram = imputed_x_data[Hist70Features]
train_x_no_histogram = imputed_x_data.drop(Hist70Features, axis=1)
# print(train_x_no_histogram.shape)
# print(imputed_x_histogram.shape)
# print(train_set_y.shape)

# These next steps is sorting out which Features are actually relevant. Using Recursive Feature Elimination, feature that are not important is removed
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from matplotlib import pyplot

rfe_selector = RFE(estimator=RandomForestClassifier(n_jobs=-1), n_features_to_select=15, verbose=5)
rfe_selector.fit(imputed_x_histogram, train_set_y)
print_header("Top 15 Features:")
print(imputed_x_histogram.columns[rfe_selector.get_support()])
top_15_feature_columns = imputed_x_histogram.columns[rfe_selector.get_support()]
dftop15 = pd.DataFrame(dataframe[top_15_feature_columns])
dfclass = pd.DataFrame(dataframe['class'])
top_15_features_data = dfclass.join(dftop15)
top_15_features_data.to_csv('data/aps_failure_top_15_features_data.csv', index=False)
print_header("Top 15 Features full training dataset")
print(top_15_features_data)

test_top15 = pd.DataFrame(test_data[top_15_feature_columns])
test_class = pd.DataFrame(test_data['class'])
top15_test = test_class.join(test_top15)
print_header("Top 15 features test dataset head:")
print(top15_test.head())
top15_test.to_csv('Data/aps_failure_test_top15.csv')

# add this to the next code:
# create new dataframe after feature selection that only have the features selected, and save to a new csv for processing.
#
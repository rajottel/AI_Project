from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sb
import sklearn
from sklearn.preprocessing import LabelEncoder
import numpy as np

df_training = pd.read_csv("Data/aps_failure_training_set.csv")
df_test = pd.read_csv("Data/aps_failure_test_set.csv")

LE = LabelEncoder()
df_training['class'] = LE.fit_transform(df_training['class'])
df_test['class'] = LE.fit_transform(df_test['class'])

df_new = df_training.replace('na', np.nan)

# count_nan = {i: list(df_training.isna().sum()*100/df_training.shape[0])[j] for j, i in enumerate(df_training.columns)}
# count_nan = {i: n for i, n in sorted(count_nan.items(), key=lambda item: item[1], reverse=True)}
#
# sb.set_style(style="whitegrid")
# plt.figure(figsize=(20,10))
#
# barplot = sb.barplot(x=list(count_nan.keys())[:15], y=list(count_nan.values())[:15], palette="hls")

# Creating a dictionary whose keys are the column names and values are the percentage of missing values
nan_count = {k:list(df_new.isna().sum()*100/df_new.shape[0])[i] for i,k in enumerate(df_new.columns)}

# Sorting the dictionary in descending order based on the percentage of missing values
nan_count = {k: v for k, v in sorted(nan_count.items(), key=lambda item: item[1],reverse=True)}
# Plotting a graph showing the top 15 features having highest percentage of missing values
sb.set_style(style="whitegrid")
plt.figure(figsize=(20,10))

# Bar Plot
plot = sb.barplot(x= list(nan_count.keys())[:15],y = list(nan_count.values())[:15],palette="hls")

# Add annotations above each bar signifying their value
for p in plot.patches:
        plot.annotate('{:.1f}%'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+1))

# Make y-axis more interpretable
plot.set_yticklabels(map('{:.1f}%'.format, plot.yaxis.get_majorticklocs()))
print(plt.show())
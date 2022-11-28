import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# This part of the code gives a plot of each identifiers and how many bins they have
dataframe = pd.read_csv("aps_failure_training_imputed.csv")
count = 0
first = []
first = dataframe.columns.str.split('_').str[0].tolist()
first.pop(0)
column_names = dataframe.columns.tolist()
column_names.pop(0)
list_of_indicators = sorted(set(first))
i=0
count_first = []
for j in list_of_indicators:
    count_first.append(first.count(list_of_indicators[i]))
    i += 1
print(list_of_indicators)
print(count_first)
plt.figure(figsize=(24,5))
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
print("These are the indicators with 10 bins each: ",HistIdenWith10Bins)

Hist70Features = []
index70F = []
g = 0
n=0
for j in first:
    for g in HistIdenWith10Bins:
        if j == g:
            index70F.append(n)
    n += 1

for j in index70F:
    Hist70Features.append(column_names[j])
print("These are the the histogram and bins: ", Hist70Features)

# imputed_histogram =
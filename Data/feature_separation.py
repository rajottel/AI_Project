import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sb
import matplotlib.pyplot as plt
#from skimpy import skim
#from sklearn.impute import KNNImputer

dataframe = pd.read_csv("aps_failure_training_imputed.csv")
count = 0
first = []
first = dataframe.columns.str.split('_').tolist()
i=0
count_first = []
for col in dataframe.columns:
    #if col.startswith(first[i]):
        #count += 1
    count = Counter(first)
    #else:
        #count_first.append(count)
        #i += 1
        #count = 0

first.pop(0)
#dataframe.columns.str
print(first)
print(list(count.values()))
plt.figure(figsize=(24,5))
sb.barplot(x=first, y=list(count.values()))
plt.show()
#print(dataframe.head())
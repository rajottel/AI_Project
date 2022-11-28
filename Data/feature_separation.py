import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

dataframe = pd.read_csv("aps_failure_training_imputed.csv")
count = 0
first = []
first = dataframe.columns.str.split('_').str[0].tolist()
first.pop(0)
list_of_indicators =sorted(set(first))
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
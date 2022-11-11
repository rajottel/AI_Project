from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sb
import sklearn
from sklearn.preprocessing import LabelEncoder

df_training = pd.read_csv("Data/aps_failure_training_set.csv")
df_test = pd.read_csv("Data/aps_failure_test_set.csv")

LE = LabelEncoder()
df_training['class'] = LE.fit_transform(df_training['class'])
df_test['class'] = LE.fit_transform(df_test['class'])

count_nan = {i:list(df_training.isna().sum()*100/df_training.shape[0])[j] for j,i in enumerate(df_training.columns)}
count_nan = {i: n for i, n in sorted(count_nan.items(), key=lambda: item[1], reverse=True)}

sb.set_style(style="whitegrid")
plt.figure(figsize=(20,10))

barplot = sb.barplot(x= list(count_nan.keys())[:15],y = list(count_nan.values())[:15], palette="hls")

plt.show()
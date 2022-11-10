from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sb
import sklearn
from sklearn.preprocessing import LabelEncoder

df_training = pd.read_csv ("Data/aps_failure_training_set.csv")
df_test = pd.read_csv ("Data/aps_failure_test_set.csv")

LE = LabelEncoder()
df_training['class'] = LE.fit_transform(df_training['class'])
df_test['class'] = LE.fit_transform(df_test['class'])
print(df_training['class'].unique())
#df_training['class'] = get_correct_label(df_training['class'])
#print(df_training['class'].unique())
print(df_training['class'].value_counts())

sb.barplot(data=df_training, x=df_training['class'].unique(), y=df_training['class'].value_counts())
plt.title('Class Distribution for Positive and Negative')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.show()
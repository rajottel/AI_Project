from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sb
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFE
from sklearn.preprocessing import LabelEncoder


df_training = pd.read_csv ("Data/aps_failure_training_set.csv")
df_test = pd.read_csv ("Data/aps_failure_test_set.csv")

LE = LabelEncoder()
df_training['class'] = LE.fit_transform(df_training['class'])
df_test['class'] = LE.fit_transform(df_test['class'])
df_new = df_training.replace('na', np.nan)

print (df_new.head())
#,'ee_001','ee_002','ee_003','ee_004','ee_005','ee_006','ee_007','ee_008','ee_009'
xs = np.arange(60000)
series1 = np.array(df_new['ee_000']).astype(np.double)
s1mask = np.isfinite(series1)
#series2 = np.array(df_new['ee_008']).astype(np.double)
#s2mask = np.isfinite(series2)

plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o')
#plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker='o')

plt.show()

#sb.heatmap(df_training.corr(), annot=True)
#plt.title("Correlation Matrix")
#plt.show()
#df_new_2 = df_new.astype(int)
#plt.hist(df_new_2['ee_000'])
from os import listdir

import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import pyplot

df_test = pd.read_csv("Data/aps_failure_test_set.csv")
df_train = pd.read_csv("Data/aps_failure_training_set.csv")

def get_correct_label(y):
      return y.replace(['neg','pos'],[0,1])

print(df_test['class'].unique())
df_test['class'] = get_correct_label(df_test['class'])
print(df_test['class'].unique())

#df_test.boxplot()

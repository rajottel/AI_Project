import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sb
import matplotlib.pyplot as plt
from skimpy import skim
test_data = pd.read_csv("data/aps_failure_test_set.csv")
train_data = pd.read_csv("data/aps_failure_training_set.csv")
# print("Test data shape:", test_data.shape)
# print("Test data head:", test_data.head())
# print("\n")
# print("Training data shape: ", train_data.shape)
# print("training data head: ", train_data.head())

labelencoder = LabelEncoder()
test_data['class'] = labelencoder.fit_transform(test_data['class'])
train_data['class'] = labelencoder.fit_transform(train_data['class'])

print("Test ['Neg'  'Pos'] =", test_data['class'].unique())
print("Training ['neg' 'pos'] =", train_data['class'].unique())

sb.pairplot(train_data, vars=['aa_000', 'ab_000', 'ac_000', 'ad_000', 'ae_000', 'af_000', 'ag_000'])
plt.show()
#skim(train_data)

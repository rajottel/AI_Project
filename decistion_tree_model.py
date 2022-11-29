import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('Data/aps_failure_top_15_features_data.csv')
dataframe["class"] = dataframe['class'].map({'neg':0,'pos':1})
df_sample = dataframe.sample(frac=0.15)
X = df_sample.drop('class', axis=1)
y = df_sample[['class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
dec_tree = DecisionTreeClassifier()

print(cross_val_score(dec_tree,X_train, y_train, scoring='f1',cv=7))
mean_score = cross_val_score(dec_tree, X_train, y_train, scoring='f1', cv=7).mean()
std_score = cross_val_score(dec_tree, X_train, y_train, scoring='f1', cv=7).std()
print(mean_score)
print(std_score)

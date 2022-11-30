import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
# From line 18-32 reference -> https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/
dataframe = pd.read_csv('Data/aps_failure_top_15_features_data.csv')
dataframe["class"] = dataframe['class'].map({'neg':0,'pos':1})
df_sample = dataframe.sample(frac=0.34)
X = df_sample.drop('class', axis=1)
y = df_sample[['class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
dec_tree = DecisionTreeClassifier()
dec_tree = dec_tree.fit(X_train, y_train)
y_pred = dec_tree.predict(X_test)
conf_m = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
ax.matshow(conf_m, cmap=plt.cm.Greens, alpha=0.3)
for i in range(conf_m.shape[0]):
    for j in range(conf_m.shape[1]):
        ax.text(x=j, y=i, s=conf_m[i,j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Confusion Matrix')
plt.show()
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

# print(cross_val_score(dec_tree,X_train, y_train, scoring='f1',cv=7))
# mean_score = cross_val_score(dec_tree, X_train, y_train, scoring='f1', cv=7).mean()
# std_score = cross_val_score(dec_tree, X_train, y_train, scoring='f1', cv=7).std()
# print("\nAverage Accuracy of Decision Tree Analysis with 7 tries: ", mean_score)
# print("Standard Deviation of the analysis: ", std_score)

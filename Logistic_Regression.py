import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Lines 1-8 from: https://www.datacamp.com/tutorial/understanding-logistic-regression-python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#Line 10 from: https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56

df_lr = pd.read_csv("Data/aps_failure_top_15_features_data.csv")
le = LabelEncoder()
df_lr['class'] = le.fit_transform(df_lr['class'])

df_lr1 = df_lr.sample(frac=0.15)

print(df_lr.head())
# print(df_lr.shape)
# print(df_lr1.head())
# print(df_lr1.shape)

# X = df_lr.drop('class', axis=1)
# y = df_lr['class']
X = df_lr1.drop('class', axis=1)
y = df_lr1['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

lr = LogisticRegression(solver='lbfgs', max_iter=10000)
#Logistic Regression code adapted from in-class lectures and lab exercises (SMRTTECH 4AI3 - Lab 5)

# fit the model with data
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Results for Logistic Regression on Sampled Dataset:')
print("------------------------------------------------")

lr.score(X_test, y_test)

#confusion matrix code adapted from: https://www.datacamp.com/tutorial/understanding-logistic-regression-python
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
plt.figure(figsize=(15,10))
sb.set(font_scale=1.4)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sb.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap=plt.cm.Greens ,linewidths=0.2)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix for Logistic Regression on Sampled Dataset')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# matrix = confusion_matrix(y_test, y_pred)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
# # Build the plot
# plt.figure(figsize=(16,7))
# sb.set(font_scale=1.4)
# sb.heatmap(matrix, annot=True, annot_kws={'size':10}, cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
# class_names = [0,1]
# tick_marks = np.arange(len(class_names))
# tick_marks2 = tick_marks + 0.5
# plt.xticks(tick_marks, class_names, rotation=0)
# plt.yticks(tick_marks2, class_names, rotation=0)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Logistic Regression on Full Dataset')
# plt.show()


print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred))
print("Recall: ", metrics.recall_score(y_test, y_pred))
print("Confusion Matrix:")
print(cnf_matrix)
plt.show()
# lr.fit(X_train, y_train)



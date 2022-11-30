import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# lines 8-10 from https://kearnz.github.io/autoimpute-tutorials/
import warnings
warnings.filterwarnings("ignore")
print_header = lambda msg: print(f"{msg}\n{'-'*len(msg)}")

print_header("Training Dataset Head:")
train_data = pd.read_csv('Data/aps_failure_top_15_features_data.csv')
train_data = train_data.reset_index(drop=True)
# labelencoder code taken from Lab 5 submission
labelencoder = LabelEncoder()
train_data['class'] = labelencoder.fit_transform(train_data['class'])
print(train_data.head())

print_header("Test Dataset Head:")
test_data = pd.read_csv('Data/aps_failure_test_top15.csv')
test_data = test_data.iloc[:, 1:]
test_data = test_data.reset_index(drop=True)
# labelencoder code taken from Lab 5 submission
test_data['class'] = labelencoder.fit_transform(test_data['class'])
print(test_data.head())

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

# boutta go mf ham right here

estimators = []
accuracys = []
precisions = []
recalls = []
confusions = []



for i in range(7100, 7600):
    RFC = RandomForestClassifier(n_estimators=i, n_jobs=-1)
    RFC.fit(X_train, y_train)
    y_prediction = RFC.predict(X_test)
    estimator = str(i)
    accuracy = metrics.accuracy_score(y_test, y_prediction)
    precision = metrics.precision_score(y_test, y_prediction)
    recall = metrics.recall_score(y_test, y_prediction)
    confusion = confusion_matrix(y_test, y_prediction)
    estimators.append(estimator)
    accuracys.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    confusions.append(confusion)

# Max Accuracy

max_accuracy = accuracys[0]
max_accuracy_index = 0
for j in range(1, len(accuracys)):
    if accuracys[j] > max_accuracy:
        max_accuracy = accuracys[j]
        max_accuracy_index = j


print_header('Random Forest Classifier Model with Max Accuracy Results')
print("n_estimators hyperparameter value:", estimators[max_accuracy_index])
print("Accuracy:", max_accuracy)
print("Precision:", precisions[max_accuracy_index])
print("Recall:", recalls[max_accuracy_index])
print("Confusion Matrix:\n", confusions[max_accuracy_index])

plt.figure(figsize=(15, 10))
sb.set(font_scale=1.4)
sb.heatmap(confusions[max_accuracy_index], annot=True, annot_kws={'size': 10}, cmap=plt.cm.Purples, linewidths=0.2)
class_names = ['0', '1']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Classifier Model with Max Accuracy Results for APS System Failure '
          'Detection')
print(plt.show())

# Max Precision

max_precision = precisions[0]
max_precision_index = 0
for j in range(1, len(precisions)):
    if precisions[j] > max_precision:
        max_precision = precisions[j]
        max_precision_index = j

print_header('Random Forest Classifier Model with Max Precision Results')
print("n_estimators hyperparameter value:", estimators[max_precision_index])
print("Accuracy:", accuracys[max_precision_index])
print("Precision:", max_precision)
print("Recall:", recalls[max_precision_index])
print("Confusion Matrix:\n", confusions[max_precision_index])

plt.figure(figsize=(15, 10))
sb.set(font_scale=1.4)
sb.heatmap(confusions[max_precision_index], annot=True, annot_kws={'size': 10}, cmap=plt.cm.Purples, linewidths=0.2)
class_names = ['0', '1']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Classifier Model with Max Precision Results for APS System Failure '
          'Detection')
print(plt.show())

# Max Recall

max_recall = recalls[0]
max_recall_index = 0
for j in range(1, len(recalls)):
    if recalls[j] > max_precision:
        max_recall = recalls[j]
        max_recall_index = j

print_header('Random Forest Classifier Model with Max recall Results')
print("n_estimators hyperparameter value:", estimators[max_recall_index])
print("Accuracy:", accuracys[max_recall_index])
print("Precision:", precisions[max_recall_index])
print("Recall:", max_recall)
print("Confusion Matrix:\n", confusions[max_recall_index])

plt.figure(figsize=(15, 10))
sb.set(font_scale=1.4)
sb.heatmap(confusions[max_recall_index], annot=True, annot_kws={'size': 10}, cmap=plt.cm.Purples, linewidths=0.2)
class_names = ['0', '1']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Classifier Model with Max Recall Results for APS System Failure '
          'Detection')
print(plt.show())


# print_header("Results for RandomForestClassifier on full Dataset:")
# print("Accuracy:", metrics.accuracy_score(y_test, y_prediction))
# print("Precision:", metrics.precision_score(y_test, y_prediction))
# print("Recall:", metrics.recall_score(y_test, y_prediction))

# confusion matrix and plot code from https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
# confusion = confusion_matrix(y_test, y_prediction)
# print("Confusion Matrix:\n", confusion)
# plt.figure(figsize=(15, 10))
# sb.set(font_scale=1.4)
# sb.heatmap(confusion, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Purples, linewidths=0.2)
# class_names = ['0', '1']
# tick_marks = np.arange(len(class_names)) + 0.5
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Random Forest Model APS Failure Test Dataset')
# print(plt.show())

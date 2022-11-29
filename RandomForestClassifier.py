import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# lines 8-10 from https://kearnz.github.io/autoimpute-tutorials/
import warnings
warnings.filterwarnings("ignore")
print_header = lambda msg: print(f"{msg}\n{'-'*len(msg)}")


dataset = pd.read_csv('Data/aps_failure_top_15_features_data.csv')
print_header("Dataset before random sampling")
print(dataset.head())
print(dataset.shape)
# labelencoder code taken from Lab 5 submission
labelencoder = LabelEncoder()
dataset['class'] = labelencoder.fit_transform(dataset['class'])
print(dataset.head())
# RandomForest code below from https://www.datacamp.com/tutorial/random-forests-classifier-python
X = dataset.drop('class', axis=1)
y = dataset['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
# clf = svm.SVC(kernel='linear')
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print_header("Results for RandomForestClassifier on full Dataset:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
# confusion matrix code from https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
full_confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", full_confusion)
plt.figure(figsize=(15, 10))
sb.set(font_scale=1.4)
sb.heatmap(full_confusion, annot=True, annot_kws={'size':10}, cmap=plt.cm.Greens, linewidths=0.2)
class_names = ['0','1']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model on full dataset')
print(plt.show())
# random sample code from https://datatofish.com/random-rows-pandas-dataframe/
dataset1 = dataset.sample(frac=0.15)
print_header("Dataset after random sampling")
print(dataset1.head())
print(dataset1.shape)
# RandomForest code below from https://www.datacamp.com/tutorial/random-forests-classifier-python
X1 = dataset1.drop('class', axis=1)
y1 = dataset1['class']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=10)
# clf = svm.SVC(kernel='linear')
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X1_train, y1_train)
y1_pred = clf.predict(X1_test)
print_header("Results for RandomForestClassifier on sampled Dataset:")
print("Accuracy:", metrics.accuracy_score(y1_test, y1_pred))
print("Precision:", metrics.precision_score(y1_test, y1_pred))
print("Recall:", metrics.recall_score(y1_test, y1_pred))
# confusion matrix code from https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
sample_confusion = confusion_matrix(y1_test, y1_pred)
print("Confusion Matrix:\n", sample_confusion)
plt.figure(figsize=(15, 10))
sb.set(font_scale=1.4)
sb.heatmap(sample_confusion, annot=True, annot_kws={'size':10}, cmap=plt.cm.Greens, linewidths=0.2)
class_names = ['0','1']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model on sampled dataset')
print(plt.show())
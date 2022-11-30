# SMRTTECH 4AI3 - Final Project
## APS System Failure Dataset
*****
## Code Developed by:
### Luc Rajotte (rajottel@mcmaster.ca)
### Daniel Sioldea (sioldead@mcmaster.ca)
### Severin Hidajat (hidajats@mcmaster.ca)
*****
As one of the final major projects of our undergraduate career, the purpose of this project is to train a machine learning model of our choice using the APS System Failure dataset. This dataset consists of a number entries falling into either the positive or negative class. The dataset has 171 features. Our team performed feature selection, data imputation, and applied three different ML models on a subste of the data to select the best model. 

The model we selected is the Random Forest Classifier. We ran through multiple iterations for the n_estimators hyperparameter to find the best accuracy, precision, and recall results within a set range. The top performer for all 3 result categories were found, and the results were displayed for those iterations. Finally, a confusion matrix plot was created for each of the top performing model iterations.

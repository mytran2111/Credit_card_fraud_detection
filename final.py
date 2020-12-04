#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf 
from sklearn.decomposition import PCA 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# Models libraries 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
import collections 

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from scipy.stats import norm

#SMOTE (over-sampling method )
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Read data
data = pd.read_csv('data/creditcard.csv')
print(data.head())

# Check the data 
print("Credit card Fraud Detection data contains rows:", data.shape[0], "columns:", data.shape[1])


# Visulazing data 
sns.countplot('Class', data=data, palette="Set3")
plt.title('Class Distributions \n (0: Genuine || 1: Fraud)', fontsize=14)


# Feature selection - by creating a drop list
drop_list_1 = ['V28','V27','V26','V25','V24','V23','V22', 'V20', 'V15', 'V13']
drop_list_2 = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
drop_list_3 = []
drop_list_4 = []



# Split data into training, validation, test set
def split_data(data, drop_list):
    data = data.drop(drop_list, axis=1)
    y = data["Class"].values # our labels
    X = data.drop(["Class"] ,axis=1).values # our features
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state=42, stratify=y) 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify=y)
    print("The number of fraud in test-set: ", sum(y_test))
    print("The size of training set : ", len(y_train),"\nThe size of test set: ", len(y_test))
    return X_train, X_test, y_train, y_test


# Get our training, validation, test set
X_train_1, X_test_1, y_train_1, y_test_1 = split_data(data, drop_list_1)
X_train_2, X_test_2, y_train_2, y_test_2 = split_data(data, drop_list_2)
X_train_3, X_test_3, y_train_3, y_test_3 = split_data(data, drop_list_3)
X_train_4, X_test_4, y_train_4, y_test_4 = split_data(data, drop_list_4)

# Correlation matrices 
# Variables with negative correlation
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=data, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=data, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=data, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

# Variables with positive correlation
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=data, ax=axes[0])
axes[0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V2", data=data, ax=axes[2])
axes[2].set_title('V2 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V19", data=data, ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')


# Anomaly Detection  
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v12_fraud_dist = data['V12'].loc[data['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='powderblue')
ax2.set_title('V12 Fraud Transaction Distribution', fontsize=14)


v10_fraud_dist = data['V10'].loc[data['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='orchid')
ax3.set_title('V10 Fraud Transaction Distribution', fontsize=14)
 

# Classification methods 

# Naive Bayes 
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
grid_naive_bayes = GridSearchCV(estimator=GaussianNB(),param_grid=params_NB,scoring='accuracy')
#naive_bayes = GaussianNB()
# Training data 
grid_naive_bayes.fit(X_train_1,y_train_1)
# Predict data 
#y_pred_naive = naive_bayes.predict(X_test_1)
# Get best parameter 
best_param = grid_naive_bayes.best_estimator_
print(best_param)

#Train our data based on the Grid Search
model = GaussianNB(var_smoothing= 2.0)
model.fit(X_train_1,y_train_1)
y_pred_naive = model.predict(X_test_1)

# Logistic Regression
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10]}
grid_log_reg = GridSearchCV(LogisticRegression(max_iter = 10000,solver = 'saga'), log_reg_params)
grid_log_reg.fit(X_train_2, y_train_2)
log_reg = grid_log_reg.best_estimator_
log_reg = LogisticRegression(max_iter = 490)
log_reg.fit(X_train_2,y_train_2)
y_pred_log_reg = log_reg.predict(X_test_2)
print(log_reg)

# Logistic Regression Grid Search 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# Get the best parameters.
log_reg = grid_log_reg.best_estimator_

# Supporting Vector Machine with Grid Search 
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train_4, y_train_4)
# get the best parameter
svc = grid_svc.best_estimator_
svc.fit(X_train_4,y_train_4)
y_pred_sup_vect = svc.predict(X_test_4)

# Random Forest 
random_forest = RandomForestClassifier(n_jobs=-1)
random_forest.fit(X_train_4,y_train_4)
y_pred_random_forest = random_forest.predict(X_test_4)


# Print scores 
def print_scores(y_test,y_pred):
    print('The confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    print("recall score: ", recall_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    print("f1 score: ", f1_score(y_test,y_pred))
    
print("Naive Bayes Gaussian Distribution results: ")
print_scores(y_test_1, y_pred_naive)
print("Logistics Regression results: ")
print_scores(y_test_2, y_pred_log_reg)
print_scores(y_train_3, y_pred_svm)

# Get confusion matrix 
naive_bayes_confusion = confusion_matrix(y_test_1, y_pred_naive)
log_reg_confusion = confusion_matrix(y_test_2, y_pred_log_reg)
sup_vect_confusion = confusion_matrix(y_test_4, y_pred_sup_vect)
rand_forest_confusion = confusion_matrix(y_test_4,y_pred_random_forest)

#Naive Bayes
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(naive_bayes_confusion, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
plt.title('Naive Bayes Classification Confusion Matrix')
plt.xlabel('y_predict')
plt.ylabel('y_test')
plt.show()
#Logistics Regression
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(log_reg_confusion, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('y_predict')
plt.ylabel('y_test')
plt.show()

# Supporting Vector 
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(sup_vect_confusion, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
plt.title('Supporting Vector Confusion Matrix')
plt.xlabel('y_predict')
plt.ylabel('y_test')
plt.show()

# Random Forest 
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(rand_forest_confusion, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
plt.title('Random Forest Confusion Matrix')
plt.xlabel('y_predict')
plt.ylabel('y_test')
plt.show()

# Look at xgboost and lightgbm 
# make a copy of X_train 
smote = SMOTE(random_state=20, n_jobs=-1)
X_train_copy = X_train.copy()
X_train_smote,Y_train_smote = smote.fit_sample(X_train_scald,Y_train)

# XGboost using SMOTE
xgboost = XGBClassifier(tree_method='gpu_hist',n_jobs=-1)
xgboost.fit(X_train_4,y_train_4)
y_xgboost = xgboost.predict(X_test_4)

xgb_smote = XGBClassifier(tree_method='gpu_hist',n_jobs=-1)
xgb_smote.fit(X_train_smote,Y_train_smote)
y_xgb_smote = xgb_smote.predict(X_train_smote)


# In[ ]:





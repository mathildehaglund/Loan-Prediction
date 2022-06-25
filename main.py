import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

train_data = pd.read_csv("train.csv")  # Reads the necessary data.
test_data = pd.read_csv("test.csv")

train_data.Loan_Status = train_data.Loan_Status.map({'Y': 1, 'N': 0})  # Approval is 1 and rejection is 0

#  print(train_data.head())
#  print(train_data.describe())

#  print(train_data.isnull().sum())

Loan_Status = train_data.Loan_Status
train_data.drop('Loan_Status', axis=1, inplace=True)
Loan_ID = test_data.Loan_ID
data = train_data.append(test_data)
#  print(data.head())
#  print(data.shape)
#  print(data.describe())

#  print(data.isnull().sum())

data.Gender = data.Gender.map({'Male':1, 'Female':0})  # Preparing data to make prediction model
#  print(data.Gender.value_counts())
data.Married = data.Married.map({'Yes':1, 'No':0})
#  print(data.Married.value_counts())
data.Dependents = data.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})
#  print(data.Dependents.value_counts())
data.Education = data.Education.map({'Graduate': 1, 'Not Graduate': 0})
#  print(data.Education.value_counts())
data.Self_Employed = data.Self_Employed.map({'Yes':1, 'No': 0})
#  print(data.Self_Employed.value_counts())
data.Property_Area = data.Property_Area.map({'Urban':2, 'Semiurban': 1, 'Rural': 0})
print(data.Property_Area.value_counts())

#  Filling missing values

data.Credit_History.fillna(1, inplace=True)
data.Married.fillna(1, inplace=True)
data.LoanAmount.fillna(data.LoanAmount.median(), inplace=True)
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(), inplace=True)
data.Gender.fillna(1, inplace=True)
data.Dependents.fillna(0, inplace=True)
data.Self_Employed.fillna(0, inplace=True)

#  print(data.isnull().sum())

data.drop('Loan_ID', inplace=True, axis=1)  # No need for the Loan_ID data

#  Splitting the data into training and testing

train_X = data.iloc[:614, ]  # all the data in the train set
train_y = Loan_Status

train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, random_state=0)

#  print(train_X.head())

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Linear Discrimant Analysis', LinearDiscriminantAnalysis()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('Support Vector Classifier', SVC()))
models.append(('K- Neirest Neighbour', KNeighborsClassifier()))
models.append(('Naive Bayes', GaussianNB()))

scoring = 'accuracy'
result = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, train_X, train_y, cv=kfold, scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print(model)
    print('%s %f' % (name, cv_result.mean()))

LR = LogisticRegression()
LR.fit(train_X, train_y)
pred = LR.predict(test_X)
print('Model Accuracy:- ', accuracy_score(test_y, pred))
print(confusion_matrix(test_y, pred))
print(classification_report(test_y, pred))

#  Saves file as pickle
file = './Model/ML_Model1.pkl'
with open(file, 'wb') as f:
    pickle.dump(f)




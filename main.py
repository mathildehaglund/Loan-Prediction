import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from joblib import dump, load

st.title("Loan prediction")

st.write("""
        This website can be used to predict if a loan application gets
        accepted or denied, using financial data as input, and a machine
        learning model as a prediction tool.""")

train_data = pd.read_csv("train.csv")  # Reads the necessary data.
test_data = pd.read_csv("test.csv")
# print(train_data.head())

train_data['Gender'] = train_data['Gender'].map({'Male': 0, 'Female': 1})
train_data['Married'] = train_data['Married'].map({'No': 0, 'Yes': 1})
train_data['Loan_Status'] = train_data['Loan_Status'].map({'N': 0, 'Y': 1})

#  print(train_data.isnull().sum())
print(test_data.isnull().sum())

train_data = train_data.dropna()
#  print(train_data.isnull().sum())

#  Seperating the dependent (Loan_Status) and the independent variables.

X = train_data[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = train_data.Loan_Status
print(X.shape, y.shape)

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.2, random_state=10)

model = RandomForestClassifier(max_depth=4, random_state=10)
model.fit(x_train, y_train)

pred_cv = model.predict(x_cv)
print(accuracy_score(y_cv, pred_cv))

pred_train = model.predict(x_train)
print(accuracy_score(y_train, pred_train))

pickle_out = open('classifier.pkl', mode='wb')
pickle.dump(model, pickle_out)
pickle_out.close()


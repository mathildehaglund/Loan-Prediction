from copy import deepcopy
import pandas as pd
import streamlit as st

st.title("Loan prediction")

st.write("""
        This website can be used to predict if a loan application gets
        accepted or denied, using financial data as input, and a machine
        learning model as a prediction tool.""")

train_data = pd.read_csv("train.csv")  # Reads the necessary data.
test_data = pd.read_csv("test.csv")
full_data = pd.concat([train_data, test_data])

pred_df = pd.DataFrame(index=test_data["Loan_ID"], columns=["Loan_Status"])  # Prepare df for future predictions.

loan_id_test_data = deepcopy(test_data["Loan-ID"])  # Saves a backup of the loan IDs.

train_data.drop("Loan_ID", axis=1, inplace=True)  # Removes loan IDs from the datasets to avoid overfitting.
test_data.drop("Loan_ID", axis=1, inplace=True)

test_data.drop("Loan_Status", axis=1, inplace=True)  # Removes loan status from test data.

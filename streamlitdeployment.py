
import pickle
import streamlit as st

model = pickle.load(open('./Model/ML_Model1.pkl', 'rb'))


def run():
    st.title('Loan Application Prediction using Machine Learning')
    account_no = st.text_input('Account number')
    full_name = st.text_input('Full name')
    choose_gen = ('Female', 'Male')
    gen_options = list(range(len(choose_gen)))
    gen = st.selectbox('Sex', gen_options, format_func=lambda x: choose_gen[x])
    mar_display = ('No', 'Yes')
    mar_options = list(range(len(mar_display)))
    mar = st.selectbox('Marital Status', mar_options, format_func=lambda x: mar_display[x])
    dep_display = ('No', 'One', 'Two', 'More than two')
    dep_options = list(range(len(dep_display)))
    dep = st.selectbox('Dependents', dep_options, format_func=lambda x: dep_display[x])
    edu_display = ('Not graduate', 'Graduate')
    edu_options = list(range(len(edu_display)))
    edu = st.selectbox('Education', edu_options, format_func=lambda x: edu_options[x])
    emp_display = ('Job', 'Business')
    emp_options = list(range(len(emp_display)))
    emp = st.selectbox('Employment status', emp_options, format_func=lambda x: emp_display[x])
    prop_display = ('Rural', 'Semi-Urban', 'Urban')
    prop_options = list(range(len(prop_display)))
    prop = st.selectbox('Property Area', prop_options, format_func=lambda x: prop_display[x])
    cred_display = ('Between 200 to 500', 'above 500')
    cred_options = list(range(len(cred_display)))
    cred = st.selectbox('Credit score', cred_options, format_func=lambda x: cred_display[x])
    mon_income = st.number_input('Applicants income ($ per month)', value=0)
    co_mon_income = st.number_input('Co-Applicants income ($ per month)', value=0)
    loan_amount = st.number_input('Loan Amount', value=0)
    dur_display = ['3 months', '6 months', '9 months', '1 year', '18 months']
    dur_options = range(len(dur_display))
    dur = st.selectbox('Loan Duration', dur_options, format_func=lambda x: dur_display)

    if st.button('Submit'):
        duration = 0
        if dur == 0:
            duration = 90
        if dur == 1:
            duration = 180
        if dur == 2:
            duration = 270
        if dur == 3:
            duration = 360
        if dur == 4:
            duration = 540
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amount, duration]]
        print(features)
        prediction = model.predict(features)
        mp = [str(i) for i in prediction]
        ans = int(''.join(mp))
        if ans == 0:
            st.error(
                'According to our predictions, you will not get a loan'
            )
        else:
            st.success(
                'According to our predictions, you will get a loan'
            )


run()




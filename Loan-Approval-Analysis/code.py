# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)

# code ends here


# --------------
# code starts here
bank.drop('Loan_ID', axis = 1, inplace = True)
banks = bank
print('Null values :\n',banks.isnull().sum())
bank_mode = banks.mode().iloc[0]
banks['Gender'].fillna(bank_mode['Gender'], inplace = True)
banks['Married'].fillna(bank_mode['Married'], inplace = True)
banks['Dependents'].fillna(bank_mode['Dependents'], inplace = True)
banks['Self_Employed'].fillna(bank_mode['Self_Employed'], inplace = True)
banks['LoanAmount'].fillna(bank_mode['LoanAmount'], inplace = True)
banks['Loan_Amount_Term'].fillna(bank_mode['Loan_Amount_Term'], inplace = True)
banks['Credit_History'].fillna(bank_mode['Credit_History'], inplace = True)
print('Banks :\n',banks)
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks,values='LoanAmount',index=['Gender','Married','Self_Employed'],aggfunc='mean')
print('Loan Amount vs Gender\n',avg_loan_amount)


# code ends here



# --------------
# code starts here
cond1 = banks['Self_Employed'] == 'Yes'
cond2 = banks['Loan_Status'] == 'Y'
loan_approved_se = len(banks[cond1 & cond2])
cond3 = banks['Self_Employed'] == 'No'
cond4 = banks['Loan_Status'] == 'Y' 
loan_approved_nse = len(banks[cond3 & cond4])
percentage_se = (loan_approved_se/614) * 100
percentage_nse = (loan_approved_nse/614) * 100
print('Percentage of loan approval for self employed people : \n',percentage_se)
print('Percentage of loan approval for people who are not self-employed :\n',percentage_nse)
# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x : x/12)
cond5 = banks[loan_term>=25]
big_loan_term = len(cond5 == True)
print('Loan amount term greater or equal to 25 years :\n',big_loan_term)


# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')['ApplicantIncome','Credit_History']
mean_values = loan_groupby.mean()
print('Income History vs Loan Amount :\n',mean_values)



# code ends here



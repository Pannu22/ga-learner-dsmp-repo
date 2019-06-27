# --------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Code starts here

# Reading Dataset
df = pd.read_csv(path)

# Storing features and target variables
X = df.drop(['customerID','Churn'], 1)
y = df['Churn']

# Splitting data into train and test
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here

# Replacing spaces with np.nan and converting dtype to float
X_train['TotalCharges'] = X_train['TotalCharges'].replace(" ", np.nan).astype('float')
X_test['TotalCharges'] = X_test['TotalCharges'].replace(" ", np.nan).astype('float')

# Replacing nan values with means and checking of nan values
X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(), inplace=True)
X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean(), inplace=True)
print('NaN values in train data :\n',X_train.isnull().sum())

# Label Encoding Categorial data
le = LabelEncoder()
X_train['gender'] = le.fit_transform(X_train['gender'])
X_train['Partner'] = le.fit_transform(X_train['Partner'])
X_train['Dependents'] = le.fit_transform(X_train['Dependents'])
X_train['PhoneService'] = le.fit_transform(X_train['PhoneService'])
X_train['MultipleLines'] = le.fit_transform(X_train['MultipleLines'])
X_train['InternetService'] = le.fit_transform(X_train['InternetService'])
X_train['OnlineSecurity'] = le.fit_transform(X_train['OnlineSecurity'])
X_train['OnlineBackup'] = le.fit_transform(X_train['OnlineBackup'])
X_train['DeviceProtection'] = le.fit_transform(X_train['DeviceProtection'])
X_train['TechSupport'] = le.fit_transform(X_train['TechSupport'])
X_train['StreamingTV'] = le.fit_transform(X_train['StreamingTV'])
X_train['StreamingMovies'] = le.fit_transform(X_train['StreamingMovies'])
X_train['Contract'] = le.fit_transform(X_train['Contract'])
X_train['PaperlessBilling'] = le.fit_transform(X_train['PaperlessBilling'])
X_train['PaymentMethod'] = le.fit_transform(X_train['PaymentMethod'])

X_test['gender'] = le.fit_transform(X_test['gender'])
X_test['Partner'] = le.fit_transform(X_test['Partner'])
X_test['Dependents'] = le.fit_transform(X_test['Dependents'])
X_test['PhoneService'] = le.fit_transform(X_test['PhoneService'])
X_test['MultipleLines'] = le.fit_transform(X_test['MultipleLines'])
X_test['InternetService'] = le.fit_transform(X_test['InternetService'])
X_test['OnlineSecurity'] = le.fit_transform(X_test['OnlineSecurity'])
X_test['OnlineBackup'] = le.fit_transform(X_test['OnlineBackup'])
X_test['DeviceProtection'] = le.fit_transform(X_test['DeviceProtection'])
X_test['TechSupport'] = le.fit_transform(X_test['TechSupport'])
X_test['StreamingTV'] = le.fit_transform(X_test['StreamingTV'])
X_test['StreamingMovies'] = le.fit_transform(X_test['StreamingMovies'])
X_test['Contract'] = le.fit_transform(X_test['Contract'])
X_test['PaperlessBilling'] = le.fit_transform(X_test['PaperlessBilling'])
X_test['PaymentMethod'] = le.fit_transform(X_test['PaymentMethod'])

y_train = y_train.replace(to_replace=['No', 'Yes'], value=[0, 1])
y_test = y_test.replace(to_replace=['No', 'Yes'], value=[0, 1])


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here

# Having look of transformed data
print('Feature train data :\n',X_train.head())
print('Feature test data :\n',X_test.head())
print('Target train data :\n',y_train.head())
print('Target test data :\n',y_test.head())

# Initialised AdaBoost model and made predictions
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)

# Checked accuracy score, confusion matrix and classification report
ada_score = accuracy_score(y_test, y_pred)
ada_cm = confusion_matrix(y_test, y_pred)
ada_cr = classification_report(y_test, y_pred)
print('Accuracy score using AdaBoost :',ada_score)
print('Confusion matrix :\n', ada_cm)
print('Classification report :\n', ada_cr)



# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here

# Initialised XGBoost and made predictions
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Calculated accuracy score, confusion matrix and classification report
xgb_score = accuracy_score(y_test, y_pred)
print('Accuracy score of XGboost :',xgb_score)
xgb_cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix of XGBoost :\n',xgb_cm)
xgb_cr = classification_report(y_test, y_pred)
print('Classification report of XGBoost :\n',xgb_cr)

# Initialised Grid Search and made predictions
clf_model = GridSearchCV(estimator=xgb_model, param_grid=parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

# Calculated accuracy score, confusion matrix and classification report
clf_score = accuracy_score(y_test, y_pred)
print('Accuracy score after grid search :',clf_score)
clf_cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix after grid search :\n',clf_cm)
clf_cr = classification_report(y_test, y_pred)
print('Classification report after grid search :\n',clf_cr)






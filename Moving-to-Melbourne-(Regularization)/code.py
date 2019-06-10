# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Code starts here
df = pd.read_csv(path)
print('Dataset :')
print(df.head())
#Storing feature variables
X = df.drop(columns = 'Price')
#Storing target variable
y = df['Price'].copy()
#Dividing into train data and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)
corr = X_train.corr()


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here

# Making a Linear Refression model
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
# Calculating R2 score
r2 = r2_score(y_test,y_pred)


# --------------
from sklearn.linear_model import Lasso

# Code starts here

# Making lasso model
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)

# Calculating R2 score
r2_lasso = r2_score(y_test,lasso_pred)
print('R2 score of lasso model :',r2_lasso)


# --------------
from sklearn.linear_model import Ridge

# Code starts here

# Making a ridge model
ridge = Ridge()
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)

# Calculating R2 score
r2_ridge = r2_score(y_test,ridge_pred)
print('R2 score of Ridge model :',r2_ridge)
# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here

#Calculating Cross Validation Score
regressor = LinearRegression()
score = cross_val_score(regressor, X_train, y_train, cv = 10)
mean_score = np.mean(score)
print('Cross Validation Score :',mean_score)
#Code ends here


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here

#Using Polynomial Regressor for making predictions
model = make_pipeline(PolynomialFeatures(2),LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Calculating R2 score
r2_poly = r2_score(y_test, y_pred)
print('R2 score of Polynomial Regressor :',r2_poly)



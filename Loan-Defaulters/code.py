# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
data = pd.read_csv(path)
X = data.drop(columns = ['customer.id','paid.back.loan'])
y = data['paid.back.loan']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid = y_train.value_counts()
label = ['Yes','No']
plt.bar(label, fully_paid)
plt.xlabel('Paid back loan')
plt.ylabel('No. of customers')
plt.show()
# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train['int.rate'] = X_train['int.rate'].map(lambda x: x.rstrip('%')).astype(float)
X_train['int.rate'] = X_train['int.rate']/100

X_test['int.rate'] = X_test['int.rate'].map(lambda x: x.rstrip('%')).astype(float)
X_test['int.rate'] = X_test['int.rate']/100

num_df = X_train.select_dtypes(np.number)
cat_df = X_train.select_dtypes('object')
print(num_df,cat_df)
# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols = list(num_df.columns)
print(len(cols))
fig, axes = plt.subplots(9, 1,figsize=(14,12))
for i in range(0,len(cols)) :
    sns.boxplot(x=y_train,y=num_df[cols[i]],ax=axes[i])
    plt.show()
# Code ends here


# --------------
# Code starts here
cols = cat_df.columns
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(14,12))

for i in range(2):
    for j in range(2):
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train, ax=axes[i,j])
        
# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# Code starts here
for i in cat_df.columns:
    X_train[i].fillna('NA')
    le =LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i].fillna('NA')
    X_test[i] = le.transform(X_test[i])
y_train.replace(to_replace ="No", value ="0",inplace=True)
y_train.replace(to_replace ="Yes", value ="1",inplace=True)
y_test.replace(to_replace ="No", value ="0",inplace=True)
y_test.replace(to_replace ="Yes", value ="1",inplace=True)
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy of Decision Tree model :',acc)
# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2 = DecisionTreeClassifier(random_state=0)
p_tree = GridSearchCV(estimator=model_2,param_grid=parameter_grid,cv=5)
p_tree.fit(X_train,y_train)
y_t_pred = p_tree.predict(X_test)
acc_2 = accuracy_score(y_test, y_t_pred)
print('Accuracy of Decision Tree after Pruning :',acc_2)
# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus
from sklearn import tree
# Code starts here
dot_data = tree.export_graphviz(decision_tree=p_tree.best_estimator_,out_file=None,class_names=['loan_paid_back_yes','loan_paid_back_no'])
graph_big = pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here



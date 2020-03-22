"""
    Multiple Linear Regression
    Backward Elimination to find features that best predict company profits.
    Author: Sola Gbenro
"""
# importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

"""
    Import the dataset
"""
dataset = pd.read_csv('50_Startups.csv')
# Matrix of independent variable
X = dataset.iloc[:, :-1].values
# Matrix of dependent variable
y = dataset.iloc[:, 4].values

# Encoding Categorical Variable
labelencoder_X = LabelEncoder()
# fitting the labelencoder over all the rows and
# the 3rd (0 indexing) column (city)
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])   
# OneHotEncoder for State. Will convert to one hot vector
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# (((AVOIDING THE DUMMY VARIABLE TRAP)))
#X = X[:, 1:]


"""
    Splitting the dataset
        
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

"""
    To find percentage difference between prediction and true values
                [ abs(V1 - V2) / ((V1 + V2)/2) ] x 100
"""
i = 0
diff_list = []
for pred in y_pred:
    abs_diff = abs(pred - y_test[i])
    per_diff = ((abs_diff)/((pred + y_test[i])/2)) * 100
    diff_list.append(per_diff)
    i += 1

total = 0
for indx in diff_list:
    total += indx

per_total = total/(len(diff_list))

# print("Overall accuracy for 'everything' model is roughly {:02.4f}%".format(100-per_total))


"""
    Building the optimal model using Backward Elimination
    
        Multiple Linear Regressino formula:
            y = b0 + b1*x1 + b2*x2 + ... + b(n)*x(n)
            where x1, x2, x(n) are the independent variables and b is the Coefficient
    
    The STATSMODELS library does not take into account a "b0" constant
    LinearRegression does, the following code is to implement manually, a "b0"
    constant.
"""
# append a column of 1s to the begining of the matrix of features
# (((THIS WILL ADD COLUMN TO END)))
# X = np.append(arr = X, values = np.ones((50, 1)).astype(int), axis = 1)
# (((THIS WILL ADD COLUMN TO BEGINNING)))
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# New OPTIMAL matrix of features
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
# change to float for STATSMODELS
X_opt = np.array(X_opt, dtype=float)

# New fit for new matrix and print out summary
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()

# We manually remove the weakest column and start the next iteration
X_opt = X[:, [0, 2, 3, 4, 5, 6]]
X_opt = np.array(X_opt, dtype=float)
# New fit for new matrix
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()

# We manually remove the weakest column and start the next iteration
X_opt = X[:, [0, 2, 4, 5, 6]]
X_opt = np.array(X_opt, dtype=float)
# New fit for new matrix
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()

# We manually remove the weakest column and start the next iteration
X_opt = X[:, [0, 4, 5, 6]]
X_opt = np.array(X_opt, dtype=float)
# New fit for new matrix
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()

# We manually remove the weakest column and start the next iteration
X_opt = X[:, [0, 4, 6]]
X_opt = np.array(X_opt, dtype=float)
# New fit for new matrix
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()

# We manually remove the weakest column and start the next iteration
X_opt = X[:, [0, 4]]
X_opt = np.array(X_opt, dtype=float)
# New fit for new matrix
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()

"""
    With this information we can see the variables with the lowest p_value
    and therefore highest statistical significance, is the Marketing followed by R&D.
    Here we compare the two models we created, the 'everything' model and 
    the model only using two variables (Marketing and R&D).
"""

dataset_2 = pd.read_csv('50_Startups.csv')
# Matrix of 2 optimal variables
X_2 = dataset.iloc[:, [0, 2]].values
# Matrix of dependent variable
y_2 = dataset.iloc[:, 4].values

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size = 0.2, random_state = 42)

# Fitting Multiple Linear Regression to the Training set
regressor_2 = LinearRegression()
regressor_2.fit(X_train_2, y_train_2)

# Predicting the Test set results
y_pred_2 = regressor_2.predict(X_test_2)

i = 0
diff_list_2 = []
for pred_2 in y_pred_2:
    abs_diff_2 = abs(pred_2 - y_test_2[i])
    per_diff_2 = ((abs_diff_2)/((pred_2 + y_test_2[i])/2)) * 100
    diff_list_2.append(per_diff_2)
    i += 1

total_2 = 0
for indx_2 in diff_list_2:
    total_2 += indx_2

per_total_2 = total_2/(len(diff_list_2))

print("Overall accuracy for 'everything' model is roughly {:02.4f}%".format(100-per_total))
print("**"*20)
print("first model accuracy {:02.4f}% optimized model accuracy {:02.4f}%"
      .format(100-per_total, 100-per_total_2))

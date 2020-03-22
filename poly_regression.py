"""
    Polynomial Regression
    Author: Sola Gbenro
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Matrix of independent variable
# Explicitily make X a matrix of 10 rows 1 column(s), otherwise default is 1d vector
X = dataset.iloc[:, 1:2].values
# DEPENDENT VARIABLE VECTOR (not a matrix, see above)
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# no split required, data is very limited

# Feature Scaling
# no feature scaling required, data is very limited

# Fitting Linear REgression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polunomial Regressino to the dataset
# Will transform X into a new matrix of features that will contrain x^2, x^3, x^n
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)

"""
    Value of X
    array([[ 1],[ 2],[ 3],[ 4],[ 5],[ 6],[ 7],[ 8],[ 9],[10]], dtype=int64)
    
    Value of X_poly
    array([[   1.,    1.,    1.,    1.],
       [   1.,    2.,    4.,    8.],
       [   1.,    3.,    9.,   27.],
       [   1.,    4.,   16.,   64.],
       [   1.,    5.,   25.,  125.],
       [   1.,    6.,   36.,  216.],
       [   1.,    7.,   49.,  343.],
       [   1.,    8.,   64.,  512.],
       [   1.,    9.,   81.,  729.],
       [   1.,   10.,  100., 1000.]])
"""
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression results
# real observation points
plt.scatter(X, y, color='red')
# predicted salaries 1-10
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
# create a 1d vector from min(X) to max(x) values are [1-10], increment by .1
X_grid = np.arange(min(X), max(X), 0.1)
# plt takes a MATRIX not a VECTOR, will require .reshape()
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
# predicted salaries 1-10
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
# takes a martrix
lin_reg.predict([[6.5]])
# output: array([330378.78787879])


# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
# output: array([133259.46969697])

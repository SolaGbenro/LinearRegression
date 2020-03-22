"""
    Decision Tree
    Author: Sola Gbenro
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Matrix of independent variable
# Explicitily make X a matrix of 10 rows 1 column(s), otherwise default is 1d vector
X = dataset.iloc[:, 1:2].values
# DEPENDENT VARIABLE VECTOR (not a matrix, see above)
y = dataset.iloc[:, 2].values

# Fitting the Decision Tree Regression model to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
# takes a martrix
y_pred = regressor.predict([[6.5]])

# Visualizing the Decision Tree Regression results IN HIGH DEFINITION
# create a 1d vector from min(X) to max(x) values are [1-10], increment by .1
X_grid = np.arange(min(X), max(X), 0.01)
# plt takes a MATRIX not a VECTOR, will require .reshape()
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
# predicted salaries 1-10
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""
    SVR
    Author: Sola Gbenro
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Matrix of independent variable
# Explicitily make X a matrix of 10 rows 1 column(s), otherwise default is 1d vector
X = dataset.iloc[:, 1:2].values
# DEPENDENT VARIABLE VECTOR (Making it a matrix, see above)
y = dataset.iloc[:, 2:3].values
# this works too
# y=y.reshape(-1,1)

# # Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting the SVR to the dataset
# sigmoid is an option here
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
# this will be an inverse of the salary as a result of the sclaing
# y_pred_2 = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualizing the SVR results
# create a 1d vector from min(X) to max(x) values are [1-10], increment by .1
X_grid = np.arange(min(X), max(X), 0.1)
# plt takes a MATRIX not a VECTOR, will require .reshape()
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
# predicted salaries 1-10
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

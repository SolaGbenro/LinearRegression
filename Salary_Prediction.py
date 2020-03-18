"""
Simple Linear Regression
Salary_Prediction
Author: Sola Gbenro
"""
# importing libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
    Import the dataset
"""
dataset = pd.read_csv('Salary_Data.csv')
# Matrix of independent variable
X = dataset.iloc[:, :-1].values
# Matrix of dependent variable
y = dataset.iloc[:, 1].values

"""
    Splitting the dataset
        
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


"""
    Fitting Simpe Linear Regression to the Training set
"""
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# visualizing the TRAINING set results
# x-axis years of experience, y-axis salary
plt.scatter(X_train, y_train, color = 'red')
# plotting against the training data NOT the test data
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualizing the TEST set results
plt.scatter(X_test, y_test, color = 'red')
# same 'simple linear regression' line, using same regressor no need to change
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

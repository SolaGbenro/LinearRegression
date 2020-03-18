"""
Simple and Multiple Linear Regression
Author: Sola Gbenro
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


"""
    Import the dataset
"""
dataset = pd.read_csv('Data.csv')
# matrix of features
# all rows, all columns but the last from dataset
X = dataset.iloc[:, :-1].values
# take all rows, and ONLY col 3 (start at 0)
y = dataset.iloc[:, 3].values

"""
    Handle missing data by replacing it with mean of column
    the missing values are np.nan objects, will become 'mean'
"""
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# fit to matrix of features 'X'
# fit along all rows, and only col 1,2 (inclusive??)
imputer = imputer.fit(X[:, 1:3])
# this is where the missing data will be transformed
X[:,1:3]=imputer.transform(X[:,1:3])

"""
    Encoding categorical data (i.e categories that are not a numerical value).
    In this example the Countries will be given unique ints to represent their
    value. France is 0 (alphabetical), Germany is 1, Spain is 2.
    
    To prevent machine learning equations from believing that Spain (current
    value of 2 which is the largest) is the best or most fit option, we will 
    create dummy encoding. Which will be similar to one-hot vectors, but 
    multiple indecies may hold value of 1 to express that more than one person
    could be from that country (i.e. four people from France).
"""
labelencoder_X = LabelEncoder()
# fitting the labelencoder over all the rows and
# the first column (country)
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])   
# OUTPUT: X[:, 0] = Array([0, 2, 1 2, 1, 0, 2, 0 , 0])

"""
    Documentation on how the column will be transformed
    ColumnTransformer(transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None,
                      transformer_weights=None, verbose=False)
    transformers:
    "List of (name, transformer, column(s)) tuples specifying the
    transformer objects to be applied to subsets of the data."
"""
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
# OUTPUT: Each row in dataset now has the 0th column (country)
# converted into a OneHotEncoding representing what country.
# index 0 is France (alphabetical) index 1 = Germany and 2 = Spain
"""
X = array([[1.0, 0.0, 0.0, 44.0, 72000.0],
           [0.0, 0.0, 1.0, 27.0, 48000.0],
           [0.0, 1.0, 0.0, 30.0, 54000.0],
           [0.0, 0.0, 1.0, 38.0, 61000.0],
           [0.0, 1.0, 0.0, 40.0, 63777.77777777778],
           [1.0, 0.0, 0.0, 35.0, 58000.0],
           [0.0, 0.0, 1.0, 38.77777777777778, 52000.0],
           [1.0, 0.0, 0.0, 48.0, 79000.0],
           [0.0, 1.0, 0.0, 50.0, 83000.0],
           [1.0, 0.0, 0.0, 37.0, 67000.0]], dtype=object)
"""
# same thing for 'Purchased' column in datasets
# transform y into array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

"""
    Splitting the dataset into the Traing set and Test set
    X_train: training part of the matrix of features (i.e. 80% of rows
                                        all columns except purchased)
    X_test: test part of the matrix of features (i.e. 20% of rows
                                        all columns except purchased)
    y_train: Indicies in the Purhcased columns matching X_train data (answers)
    y_test: Indicies in the Purchased columns matching X_test data (answers)
        
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""
    In Feature Scaling there are two main ways to standarize your data.
    Standardisation and Normalisation.
    
    X(stand) = X-mean(X) / standardDeviation(X)
    
    X(norm) = X-min(X) / max(x)-min(x)
    
    After these transforms X_train and X_test no longer have their actual values
    will now store a float between -1 and 1 (sometimes more)
"""
sc_X = StandardScaler()
# fit object to training set, then transform
X_train = sc_X.fit_transform(X_train)
# sc_X has been fit, no longer required
X_test = sc_X.transform(X_test)
# FEATURE SCALING OF DUMMY VARIABLES WOULD OCCUR HERE





























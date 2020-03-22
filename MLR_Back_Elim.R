# Multiple Linear Regression
# Sola Gbenro

# Importing the dataset
dataset = read.csv('50_Startups.csv')
# transform the dataset if required
# dataset = dataset[, 2:3]

# Encoding the categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# SPlitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# in this instance will be handled by function below 

# fitting Multiple Linear REgression to the Training set
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
# " . " tells the lm method to take all the features
regressor = lm(formula = Profit ~ .,
               data = training_set)

#  in Console  " summary(regressor) " to see the tables including p_values

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
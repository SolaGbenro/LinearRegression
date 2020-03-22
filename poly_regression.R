# Polynomial Regression
# Author: Sola gbenro

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
# indexing in R starts at 1 not 0
dataset = dataset[2:3]

# SPlitting the dataset into the Training-set and Test-set 
#install.packages('caTools')
#library(caTools)
#set.seed(123)
# only 10 observations, no need to split
#split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
#trainging_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature scaling
#training_Set = sscale(training_set)
#test_set = scale(test_set)

# Fitting the Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
# summary(lin_reg) shows p_value of .00383 for Level right now

"fitting the Polynomial Regression to the dataset
  A poly regression model is a MLR which is composed of 1 independent variable
  and additional independent variables that are polynomial terms of the first independent variable (i.e x^2, x^3, x^n" 
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

#summary(poly_reg)

# Visualizing the Linear Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression') +
  xlab('Level') +
  ylab('Salary')

# visualizing the Polynomial Regression results
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression
# prediction will be a single value
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicitng a new result with Polynomial Regression
# prediction will be a single value
y_pred_poly = predict(poly_reg, data.frame(Level = 6.5,
                                           Level2 = 6.5^2,
                                           Level3 = 6.5^3,
                                           Level4 = 6.5^4))

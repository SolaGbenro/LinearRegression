# SVR
# Author: Sola gbenro

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
# indexing in R starts at 1 not 0
dataset = dataset[2:3]

# Fitting the SVR to the dataset
#install.packages('e1071')
library(e1071)

# Fitting the Regression Model to the dataset
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')

# Predicitng a new result
# prediction will be a single value
y_pred = predict(regressor, data.frame(Level = 6.5))

# visualizing the SVR Model results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('(SVR Model)') +
  xlab('Level') +
  ylab('Salary')

# visualizing the Regression Model results IN HIGH DEFITION (i.e more predictions)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('(Regression Model)') +
  xlab('Level') +
  ylab('Salary')

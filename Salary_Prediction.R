#   Simple Linear Regression
#   Salary_Prediction
#   Author: Sola Gbenro

#   Import dataset
#   indexing starts at 1
dataset = read.csv('Salary_Data.csv')
# dataset = dataset[, 2:3]

#   Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
#   wil return true or false depending on whether data was chosen for Training (true) or not (false) 
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#   Fitting Simple Linear Regression to the Training set
#   Salary ~(proporitional) to YearsExperience
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#   Predicting the Test set results
#   vector of predictions
y_pred = predict(regressor, newdata = test_set)

#   Visualizing the Training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

#   Visualizing the Test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')

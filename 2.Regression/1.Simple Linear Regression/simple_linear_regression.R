# Data Preprocessing

# 2.Importing the dataset
dataset = read.csv('Salary_Data.csv') 

# 5. Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)     # to have same data on split
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# again feature scaling not required(inbuilt in method used)

# SIMPLE LINEAR REGRESSION

# Fitting Simple Linear Regression to the Training Set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# type summary(regressor) on console


# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualising the Training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
            colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years Of Experience') +
  ylab('Salary')
# geom stands for geometrical

# Visualising the Test set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years Of Experience') +
  ylab('Salary')

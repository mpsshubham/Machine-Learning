# Data Preprocessing

# 1. Importing Libraries
#libraries can be imported from the packages section in R

# 2.Importing the dataset
# first set the working directory(containing dataset) from R file explorer
# and then click more and then set as working directory

dataset = read.csv('Data.csv')     # select and press ctrl + enter for running
# dataset = dataset[2:3]           # to select subset of data
# R indexes starts with one unlike python
# In R we dont need to distinguish between dependent and independent variables

# 3. Taking Care Of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     	ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                    	dataset$Salary)

# 4. Encoding Categorical Data
# In R we will use factor function which will transform categorical data into
# numeric categories and will see variables as factors and we can choose labels of those factors

dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),  # c() in R stands for array/vector
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No','Yes'),  # c() in R stands for array/vector
                           labels = c(0,1))

# 5. Splitting the dataset into the Training set and Test set
# Machine Learning Models first learn by using correlation between training sets
# and then we use these models on our test set
# install.packages('caTools')
# still the package is not ticked in packages section, either do it manually
# or by writing library(package name)
library(caTools)
set.seed(123)     # to have same data on split
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# unlike python here we had to give only dependent variable(y) as argument
# and percentage of training set
# the split function return true or false for every observation
# true if it belongs to training set otherwise false
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# 6. Feature Scaling
# In machine learning we need to have our features on same scale because most of ml models use
# Euclidean distance for computation and if we have features not on same scale, then one feature 
# will dominate heavily over other
# here salary will dominate heavily over age
# there are two types of feature scaling
# 1. Standardisation x = (x - mean(x)) / standard deviation(x)
# 2. Normalisation x = (x - min(x)) / (max(x) - min(x))

# training_set = scale(training_set)
# test_set = scale(test_set)
# > training_set = scale(training_set)
# Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric
# as we saw earlier R treats encoded value as factors(not numeric)
# so we need to choose the columns which contain numeric data
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])

# Data Preprocessing
dataset = read.csv('50_Startups.csv') 

# 4. Encoding Categorical Data
dataset$State = factor(dataset$State,
                       levels = c('New York','California','Florida'),  # c() in R stands for array/vector
                       labels = c(1,2,3))

# 5. Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)     # to have same data on split
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Multiple Linear Regression
# Fitting Multiple Linear Regression to the training set

# or we can write formula = Profit ~ .
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)
# type summary(regressor) on console
# we can see that there are two dummy variables state2 and state3 which R created themselves as we encoded
# state as factors, and also R do not fell into the dummy variable trap by just creating 2 state variables
# from summary, we need to last two colmuns P value and significance 
# Lower the P value, higher the significance, *** indicates higher significance
# intercept and R&D shows lowest P value and highest significance level
# thus only R&D is strong predictor, others have almost no effect

# Predicting the test set results
y_pred = predict(regressor, newdata = test_set)

# Building The optimal model using Backward Elimination
# we can use same regressor here unlike python
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
# we have replaced training_set here by dataset(not compulsary), so that we have complete set for backward elimination
summary(regressor)

# Both state2 and state3 have very high P value, so we are removing both of them in one go

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

# Now removing Administration
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

# After removing Administration, we have a decrease in P value of Marketing from 10% to 6%, very close to our significance level
# But we still remove it 
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)

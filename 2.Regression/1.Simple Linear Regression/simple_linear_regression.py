#Data Preprocessing

# 1. Importing Libraries

import numpy as np                 # contains mathematical tools
import matplotlib.pyplot as plt    # plot charts
import pandas as pd                # for importing and managing datasets

# 2.Importing the Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,1].values  
# we can see X is matrix of features(30,1), whereas y is vector(30,)

# 5. Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 0)
# Simple Linear Regression libraries take care of feature scaling themselves, so no need of explicitly doing it

# Simple Linear Regression
# y = b0 + b1 * x1
# y - dependent variable(salary)
# x1 - independent variable(experience)
# b1 - coefficient(slope), how unit change in x1 affect y
# b0 - constant term, where line crosses vertical axis(x == 0, experience == 0)
# salary = b0 + b1 * experience
# we try to find the best line that fit our data by minimzing the sum of squared difference 
# between the observed and modelled value(Ordinary Least Square Method )
 
# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()    #press ctrl + i
# now fit this regressor model to our training set
regressor.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = regressor.predict(X_test)
# Now compare y_test(real salary) and y_pred(predicted salary)

# Visualising the Training Set Results
plt.scatter(X_train, y_train, color = 'red')    # plotting observation point(real values)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  # plotting regression line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test Set Results
plt.scatter(X_test, y_test, color = 'red')    # plotting observation point(real values)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  # plotting regression line
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

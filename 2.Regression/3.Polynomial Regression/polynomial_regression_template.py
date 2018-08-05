# Data Preprocessing

# 1. Importing Libraries
import numpy as np                 # contains mathematical tools
import matplotlib.pyplot as plt    # plot charts
import pandas as pd                # for importing and managing datasets

# 2.Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Here we need to include only level column because position and level column are equivalent
# X = dataset.iloc[:,1].values  using this we get X as a vector, not matrix of features
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:,2].values 

# Here we dont need to split the dataset into training and testing data set, as
# we hava very small dataset and we need to make very accurate predictions, so
# we cant afford to loose any single data while building model

# POLYNOMIAL REGRESSION
# y = b0 + b1x1 + b2x1^2 + ... + bnx1^n (same variable x1 in different power)
# Useful when data is not arranged in linear fashion (curves)
# Why it is called Linear????
# the linearity and non linearity in above equation is not described by x, but instead
# depends on constants b0,b1. If we can write equation by having linear combination of constants.

# We will make both linear and polynomial regressor just to compare our results

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
# poly_reg will contain feature matrix of all possible combination of the features with degree 
# less than or equal to the specified degree.
X_poly = poly_reg.fit_transform(X)
# Here X_poly contains x1 in the middle column, x1^2 in the last column, and the poly_reg object
# automatically created columns of ones(x0) as the first columns that we need in backward elimination
# for the constant b0 (for degree 2)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising Polynomial Regression Results
plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue') this would have worked fine, but if we have new X then we need to have new X_poly
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomail Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Now to improve our curve, we are changing degree from 2 to 3 and reexecuting our code
# With degree 4, we get the best curve, predicting values very close to real

# we can see from graph that it is plotting straight lines in between 2 points(1-2,2-3,3-4,...)
# so we can improve our graph by reducing this gap size by choosing much smaller values
# so we create new X_grid
X_grid = np.arange(min(X), max(X), 0.1)    # this will return vector, so convert to matrix
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomail Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# so this will result in a more continous curve

# Predicting new results using Linear Regression
lin_reg.predict(6.5)

# Predicting new results using Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

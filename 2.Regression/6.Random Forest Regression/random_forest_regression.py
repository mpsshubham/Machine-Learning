# Random Forest Regression
# Random forest is a version of ensemble learning
# ensemble learning - when you take multiple algorithm or the same algorithm multiple times  
# and we put them together to make something powerful than the original.

# 1) pick at random k data points from the training set
# 2) build the decision tree associated to these k data points
# 3) choose the number Ntrees of trees you want to build and repeat steps 1 and 2.
# 4) for a new data point, make each one of your Ntrees trees predict the value of y for the
# data point in question, and assign the new data point the average across all of the
# predicted y values.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# again random forest cant be visualised in low resolution
# with many trees we simply have more number of steps as compared to one tree
# if we add more trees its not going to add that much number of steps as after
# sometime the average would start getting converge
# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
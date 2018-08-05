# DECISION TREE REGRESSION

# CART -> Classification trees or Regression trees
# Regression trees are bit more complex than classification trees
# We can learn decision tree using one or two independent variables
# with two independent variable x1 and x2 on axis, we try to create split based on information gain and
# entropy called leaves and final leaves are called terminal leaves.
# while choosing split we draw decison tree(yes/no) type tree.
# So evaluating y for new value(x1=20,x2=50) we choose the average of all values in that terminal leaf as answer.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor =  DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# in each interval decision tree is taking average so that should be the horizontal lines
# so above diagram does not follow that(not showing constant value for a particular interval)
# all this problem is due to resolution that we picked for plotting.
# it is just predicting values for just 10 values and then just joining them with a straight line.
# previous all models were continous.Decision tree model is non continous
# the best way to visualise decision tree is high resolution.

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

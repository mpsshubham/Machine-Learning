# SVR
# Second non linear model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
# SVR class is less used and thus we need to do feature scaling explicitly, otherwise
# model built would be very bad(it was a straight line)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# here error was shown that y is a vector, so we need to transform the vector y into 
# matrix, as done above

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
# important parameter is kernel, can be linear, poly, rbf
regressor.fit(X, y)

# Predicting a new result
# as we had apllied feature scaling, so we cannot directly predict the result
# we also need to scale value 6.5
# the input to transform must be array, so we need to have an array of 6.5
# 2 [[]] are added so that 6.5 should convert into a matrix, not vector
# Now we want our prediction in original value, not scaled, so we need to do inverse transform
# using sc_y object
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# the SVR model considered the ceo level 10 as outlier point and thus is not fitting 
# the curve(model) for that point

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

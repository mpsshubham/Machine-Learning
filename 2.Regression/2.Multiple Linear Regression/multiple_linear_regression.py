# Data Preprocessing
# 1. Importing Libraries

import numpy as np                 # contains mathematical tools
import matplotlib.pyplot as plt    # plot charts
import pandas as pd                # for importing and managing datasets

# 2.Importing the Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values  
y = dataset.iloc[:,4].values 

# 4. Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]            # Removing zeroth index column(taking everything from index 1 to last)
# Many python libraries take care of this by themselves, but for some we need to explicitly do it
# This is just for demonstration purpose, python libraries take for this themselves

# 5. Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# No feature scaling required(inbuilt)

# MULTIPLE LINEAR REGRESSION
# y = b0 + b1x1 + b2x2 + .... + bnxn
# Assumptions of Linear Regression
# 1.Linearity
# 2.Homoscedasticity
# 3.Multivariate normality
# 4.Independence of errors
# 5.Lack of multicollinearirty

# Now we need to encode state column(categorical data) using dummy variable
#   y      = b0 + b1x1 +  b2x2  +   b3x3        +  b4d1
# profit          R&D    Admin    Marketing       dummy
# dummy variable acts as a switch. If it is off it works for california and if on, then for new york
# we havn't included both dummy variable(newyork and california) bcoz (d1 always = 1-d2)
# so by using just one dummy variable(new york), we can get whole info(if new york or california)
# the dummy variable not present(california) is implicitly(default) present in model equation in constant b0
# so when d1 is zero, the whole equation works for california column
# Always omit one dummy variable while building models(if 9 then use 8, if 2 then 1)
# This is called as dummy variable trap

# Refer Step-by-step-Blueprints-For-Building-Models.pdf

# Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)
# Compare y_test and y_pred

# Building the optimal model using Backward Elimination
# Useful if some independent variable are not of much statistical importance and some are of high importance
# So goal here is to have a team of independent variables of high statistical significance
# statsmodel used below do not consider the term b0 in the equation
# so we had to add a column of ones(x0) in the X matrix, so that our model does not skip the b0 term
# Without adding ones, our equation will be y = b1x1 + b2x2 +....

import statsmodels.formula.api as sm
# X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis=1)
# above call will add the column of ones at the end of X, so just change the order of first two parameters
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
# .astype(int) is used to avoid datatype error

# Backward elimination works by including all the independent variables first and then removing one by one 
# the variables(features) which are not statistically important
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# We need to explicitly write all the columns(features)
# As we are using a new library, we need to have a new regressor that will fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# OLS stans for ordinary least square, regressor used in statsmodel library
# OLS Help clearly shows that intercept is not included by default
# Backward elimination step 2 done

regressor_OLS.summary()
# lower the P value, more significant the independent variable is wrt dependent variable
# x2 has the highest P value, greater than significance level
# constant, x1, x2, x3, x4, x5
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Now remove x1
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Now remove x2
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Now index x2 is just above significance level(5%), but we had to remove it also
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# So only one independent variable(R&D spend) is left and is of most significance

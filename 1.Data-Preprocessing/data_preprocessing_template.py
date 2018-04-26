# Data Preprocessing

# 1. Importing Libraries

import numpy as np                 # contains mathematical tools
import matplotlib.pyplot as plt    # plot charts
import pandas as pd                # for importing and managing datasets

# 2.Importing the Dataset

# first set the working directory(containing dataset) from spyder file explorer  
# and save your python file in that directory and then click run(green button)
# or press F5

dataset = pd.read_csv('Data.csv')
# select the line and press ctrl + enter and check the dataset variable from variable explorer
# python indexing starts from zero

# Now we need to distinguish matrix of features(independent variables,3) and dependent variables(purchased)
X = dataset.iloc[:,:-1].values      # check by writing X in console
# left of the comma represents all the lines(rows) and the right represents columns(-1 for excluding last column)
y = dataset.iloc[:,3].values        # right side represents index of last column

# 3. Taking care of missing data
# Either remove the entry containing missing data(not good) or replace it with the mean of the columns
from sklearn.preprocessing import Imputer   # scikit learn
imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0)   # creating object of the class
# select the word(Imputer) and press ctrl+I for help
# Now we need to fit this imputer object to our matrix
imputer = imputer.fit(X[:,1:3])
# here upper bound is excluded so it will fit imputer object to column 1 and 2 only(containing missing data)
# uptill now only imputer object is transformed, X is not transformed
# Now replace missing data by mean of column
X[:,1:3] = imputer.transform(X[:,1:3])

# 4. Encoding Categorical Data(YES/NO ,Germany/France/Spain)
# Machine Learning Models deals with mathematical equations so we need to encode text into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# Using LabelEncoder we encode text to numbers but since 1>0 and 2>1 the equations in the model
# will think that spain has higher value than germany and france but thats not the case
# These are just three categories
# So we will be using dummy variables with number of columns equal to number of categories
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# We dont need to use onehotencoder for dependent variable as machine learning model will know its a category
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# 5. Splitting the dataset into the Training set and Test set
# Machine Learning Models first learn by using correlation between training sets(X_train,y_train)
# and then we use these models on our test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
# random_state is used to have same data on split(seed in R)

# 6. Feature Scaling
# In machine learning we need to have our features on same scale because most of ml models use
# Euclidean distance for computation and if we have features not on same scale, then one feature 
# will dominate heavily over other
# here salary will dominate heavily over age
# there are two types of feature scaling
# 1. Standardisation x = (x - mean(x)) / standard deviation(x)
# 2. Normalisation x = (x - min(x)) / (max(x) - min(x))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# in training set we need to use fit_transform and in test set we need to use just transform
X_test = sc_X.transform(X_test)
# its our wish if we want to scale our dummy variable or not
# we dont need to scale dependent(y) variable in classification problem but we need to scale in regression problems
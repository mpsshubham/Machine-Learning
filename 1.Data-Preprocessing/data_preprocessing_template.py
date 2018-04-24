#Data Preprocessing

#Importing Libraries

import numpy as np                 #contains mathematical tools
import matplotlib.pyplot as plt    #plot charts
import pandas as pd                #for importing and managing datasets

#Importing the Dataset

#first set the working directory(containing dataset) from spyder file explorer  
# and save your python file in that directory and then click run(green button)
# or press F5

dataset = pd.read_csv('Data.csv')
#select the line and press ctrl + enter and check the dataset variable from variable explorer
#python indexing starts from zero

#Now we need to distinguish matrix of features(independent variables,3) and dependent variables(purchased)
X = dataset.iloc[:,:-1].values      #check by writing X in console
#left of the comma represents all the lines(rows) and the right represents columns(-1 for excluding last column)
y = dataset.iloc[:,3].values        #right side represents index of last column

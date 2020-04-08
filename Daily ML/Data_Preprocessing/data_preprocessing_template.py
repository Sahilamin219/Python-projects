# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#splitting dataset into training dataset and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler(X)
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)"""
#simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting dataset into training dataset and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state =0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler(X)
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)"""

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the Test set result
y_pred  = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experince (Training set)")
plt.xlabel("Years of Experince")
plt.ylabel("Salary")
plt.show()
#Visualising the Test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experince (Test set)")
plt.xlabel("Years of Experince")
plt.ylabel("Salary")
plt.show()
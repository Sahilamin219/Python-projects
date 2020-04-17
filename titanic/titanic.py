import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train=pd.read_csv('train.csv')
# print('Train Shpae:',train.shape)
# print(train.head())
# print(train.tail())
# print(train.isnull().sum())
test=pd.read_csv('test.csv')
# print('Test Shape:',test.shape)
print('Train columns:',train.columns.tolist())
print('Test columns:',test.columns.tolist())

from sklearn.linear_model import LogisticRegression
logisticregression=LogisticRegression()
logisticregression.fit(X=pd.get_dummies(train['Sex']),y=train['Survived'])
test['Survived']=logisticregression.predict(pd.get_dummies(test['Sex']))
test[['PassengerId', 'Survived']].to_csv('kaggle_submission.csv', index = False)
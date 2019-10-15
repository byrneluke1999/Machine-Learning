import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold 

dataset = pd.read_csv('C:/Users/byrne/Desktop/Machine Learning/tcd ml 2019-20 income prediction training (with labels).csv')

'''Multiple Linear Regression'''
dataset.isnull().any()
dataset = dataset.fillna(method='ffill')


#creates a new column with numerical data for categorical data
dataset["Gender"] = dataset["Gender"].astype('category')
dataset["Gender_Cat"] = dataset["Gender"].cat.codes

dataset["University Degree"] = dataset["University Degree"].astype('category')
dataset["University_Degree_Cat"] = dataset["University Degree"].cat.codes

dataset["Profession"] = dataset["Profession"].astype('category')
dataset["Profession_Cat"] = dataset["Profession"].cat.codes

dataset["Country"] = dataset["Country"].astype('category')
dataset["Country_Cat"] = dataset["Country"].cat.codes

#dataset = pd.get_dummies(dataset, prefix_sep='_', drop_first=True)
#print(dataset.shape)

X = dataset[['Year of Record','Age', 'Body Height [cm]',
'Gender_Cat','Profession_Cat','Country_Cat']]
Y = dataset['Income in EUR'].values



X_train, X_test, Y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state = 0)



#scaler = preprocessing.MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#mm_scaler.transform(X_test)
#X_train_scaled = preprocessing.scale(X_train)
#X_test_scaled = preprocessing.scale(X_test)
#X_test_scaled = scaler.transform(X_test)

regressor = Ridge(alpha = 1000)
#regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred1 = regressor.predict(X_test)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns= ['Coefficient'])
print(coeff_df) 

rms = sqrt(mean_squared_error(y_test, y_pred1))
print(rms)

r2 = r2_score(y_test, y_pred1)
print(r2)

dataset2 = pd.read_csv('C:/Users/byrne/Desktop/Machine Learning/tcd ml 2019-20 income prediction test (without labels).csv')

dataset2.isnull().any()
dataset2 = dataset2.fillna(method='ffill')

dataset2["Gender"] = dataset2["Gender"].astype('category')
dataset2["Gender_Cat"] = dataset2["Gender"].cat.codes

dataset2["University Degree"] = dataset2["University Degree"].astype('category')
dataset2["University_Degree_Cat"] = dataset2["University Degree"].cat.codes

dataset2["Profession"] = dataset2["Profession"].astype('category')
dataset2["Profession_Cat"] = dataset2["Profession"].cat.codes

dataset2["Country"] = dataset2["Country"].astype('category')
dataset2["Country_Cat"] = dataset2["Country"].cat.codes

X1 = dataset2[['Year of Record','Age', 'Body Height [cm]',
'Gender_Cat','Profession_Cat','Country_Cat']]


Y_pred = regressor.predict(X1)
df = pd.DataFrame({'Predicted': Y_pred.flatten()})
print(df.to_csv("test5.csv"))

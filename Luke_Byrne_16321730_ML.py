import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
#from catboost import CatBoostRegressor
import xgboost as xgb

#Reading in the data initially
dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

# Catering for missing values and filling them with previous value
#dataset.isnull().any()
#dataset = dataset.fillna(method='ffill')

#I regarded 'Hair Color' , 'Size of city', 'University Degree' and 'Wears Glasses' as not relevant/important features.
X = dataset[['Year of Record', 'Age', 'Body Height [cm]',
             'Gender', 'Profession', 'Country']]
Y = dataset['Income in EUR'].values

#Splitting the data into training and testing data. X is the predictors and Y is the target that we want to predict from X.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

gsc = GridSearchCV(
    #estimator=CatBoostRegressor(),
    estimator = xgb.XGBRegressor(),
    param_grid={
        #I tried the parameters commented out here, however they had negative effects on the rmse & thus I just used the following two
        #as I found them to be the best. 
        'max_depth': range(10,20),
        'n_estimators': (300,500),
        #'colsample_bytree': [0.1],
        #'subsample': [.99],
        #'objective': ['reg:linear'],
        #'colsample_bytree': [.05],
        #'max_features': ['auto', 'sqrt', 'log2'],
        #'min_samples_split': (4,6,8),

        # # For CatBoost, the following parameters helped the score and reduced overfitting by limiting num of iterations.
        #'od_type': ["Iter"],
        #'od_wait': [100]
    },
    cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

#Preparing the features to be passed through the pipeline. Categorical & numerical features are split so as to
#encode the categorical features only. 
num_c = ['Year of Record', 'Age', 'Body Height [cm]']
cat_c = ['Gender', 'Profession', 'Country']

#Constructing the Pipeline. First I perform a Simple Imputer on the features to cater for missing values. 
#Target encoder was the best encoder for me. I  give it the index for the categorical columns [3,4,5]. These are the correct indices after
#the imputing step affects the ordering of the columns. 
#Then Gridsearch is performed. 
pi = Pipeline(steps=[('Imputer', (ColumnTransformer(transformers=[('num', SimpleImputer(strategy='mean'), num_c), ('cat', SimpleImputer(strategy = 'most_frequent', fill_value='missing'), cat_c)]))),
                     ('enc', TargetEncoder(cols=[3,4,5])),
                     ('grid', gsc)])

#fitting the pipeline
pi.fit(X_train, Y_train)

#The best parameters from Grid search for the Random Forest. 
best_params = pi._final_estimator.best_params_
print(best_params)

Y_pred1 = pi.predict(X_test)

#rmse score
rms = sqrt(mean_squared_error(Y_test, Y_pred1))
print(rms)

#r2 score
r2 = r2_score(Y_test, Y_pred1)
print(r2)

#Loading in the second dataset for fitting the model. 
dataset2 = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')


#dataset2.isnull().any()
#dataset2 = dataset2.fillna(method='ffill')

#Same Features as used to train the model. 
X1 = dataset2[['Year of Record','Age', 'Body Height [cm]',
'Gender','Profession','Country']]

#Predicting the income based on the above features.
Y_pred = pi.predict(X1)

#writing the predictions to the file.
df = pd.DataFrame({'Income': Y_pred.flatten()})
print(df.to_csv("tcd ml 2019-20 income prediction submission file(1).csv"))

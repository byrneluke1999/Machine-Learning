import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import RidgeCV, Ridge
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

#dataset.isnull().any()
#dataset = dataset.fillna(method='ffill')

X = dataset[['Year of Record', 'Age', 'Body Height [cm]',
             'Gender', 'Profession', 'Country']]
Y = dataset['Income in EUR'].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

gsc = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={
        'max_depth': range(13, 20),
        'n_estimators': (50, 100, 300),
        #'min_samples_leaf': (10,50),
        #'max_features': ['auto', 'sqrt', 'log2'],
        #'min_samples_split': (4,6,8),
    },
    cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)


num_c = ['Year of Record', 'Age', 'Body Height [cm]']
cat_c = ['Gender', 'Profession', 'Country']

pi = Pipeline(steps=[('Imputer', (ColumnTransformer(transformers=[('num', SimpleImputer(strategy='mean'), num_c), ('cat', SimpleImputer(strategy = 'most_frequent', fill_value='missing'), cat_c)]))),
                     ('enc', TargetEncoder(cols=[3,4,5])),
                     #('reduce_dims', PCA(n_components=3)),
                     ('scalar', StandardScaler()),
                     ('grid', gsc)])


pi.fit(X_train, Y_train)

best_params = pi._final_estimator.best_params_
print(best_params)

Y_pred1 = pi.predict(X_test)


rms = sqrt(mean_squared_error(Y_test, Y_pred1))
print(rms)

r2 = r2_score(Y_test, Y_pred1)
print(r2)


dataset2 = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

dataset2.isnull().any()
dataset2 = dataset2.fillna(method='ffill')

X1 = dataset2[['Year of Record','Age', 'Body Height [cm]',
'Gender','Profession','Country']]

Y_pred = pi.predict(X1)
df = pd.DataFrame({'Predicted': Y_pred.flatten()})
print(df.to_csv("preds.csv"))

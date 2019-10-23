Luke Byrne - 16321730 - D88E21A4D92AE9C27279 - 
Machine-Learning - CSU44061 - 
Trinity College Dublin - Senior Sophister - Machine Learning Course Competition Submission

The final submission makes use of the XGBoost Regressor. However, my code for CatBoost is commented out in this file as this was my second best score and wasn't far off my score using XGBoost. Some parameters are also commented out for both regressors, as these are some that I tested and usd either intiially or throughout but didn't contribute to my best score.
https://github.com/byrneluke1999/Machine-Learning

***INFORMATION FROM EXCEL SHEET***
Module Code:
CSU44061	

Course:
CS and Language

Local RMSE:
60. 339 	

ML Library:
scikit-learn	

Other Library:
xgboost	

Algorithm:
XGBoosting		

Preprocessing Libraries:
pandas, numpy, category encoders	

Feature selection Method:
Manually dropped 'Hair Color', 'Size of city', 'Wears Glasses', 'Uni degree'. After experimenting with removing different features, removing these had the best impact.

Removed Features:
Hair Color', 'Size of city', 'Wears Glasses', 'Uni degree'.

Feature Scaling:
I tried implementing a scalar using Min max scalar and standard scalar. However, this had negative effects on ny rmse score for various regressors. In my final code, a scalar isn't implemented.	

Feature Encoding:
From the category encoders library, I used Taregt encoder. This took the categorical features and replaced them with combinations of the probabilty of the target given a particular categorical variable. Before this I used oneHotEncoder with which I had a lot of trouble with improving the score further from ~80k.	

Missing Values:
At the start I used the .fillna method but after learning about pipelines, I implemented a simpleImputer with the mean so that missing values were replaced. I played around a little, swapping the mean for the median but this had little effect.	

Outlier Detection:
No outlier detection	

Data Split:
70-30 Split	

Additional Steps:
I progressed from Linear Regression to ridge regression with little change to my logic which improved my score slightly. Change to RidgeCV and playing around with alpha values improved my score. Before using TaregtEncoding, I used Binary encoding which improved my score from 140k to 120k. Using a Pipeline really boosted my score and got me in the 80,000's even using linear & ridge regression. This was the greatest improvement to my score. I learned about random Forest regression and tried implementing that within a GridsearchCV and upon first try this had negative effects on my score. After playing around and testing the effects of the different parameters, I found some that had positive impacts on my score and this boosted me to about ~72k. Branching out from scikit-learn regressors, I started experimenting with CatBoost which had positive effects on my score and brought me down to 65k.  I feared I was overfitting using this regressor, so I decided to look into xgboost. This gave me the best rmse score both locally and on kaggle. 

Most Important Steps:
I thought using PCA, I would improve my feature selection and in turn my score, but this was wrong. THis had sever effects on my score and set me back to scores of 100k. 

Additional Remarks:
It was interesting to see the effects different regressors had on the rmse, but sometimes it was difficult to understand why some were working better than others. 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 04:26:39 2019

@author: astro
"""
# %%
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

# %%
# Read csv file
train_file = '~/Documents/AIWP/Lab/LabTask_3\
/house-prices-advanced-regression-techniques\
/train.csv'
data = pd.read_csv(train_file)
summary = data.describe()

# %%
# Select features
y = data.SalePrice
data_features = [x for x in data.columns if str(data[x][0]).isdigit()][:-1]
X = data[data_features]
describe = X.describe()
head = X.head()

# %%
# Define and Fit model
data_model = DecisionTreeRegressor(random_state=1)
data_model.fit(X, y)

# %%
# Predict
test_file = '~/Documents/AIWP/Lab/LabTask_3\
/house-prices-advanced-regression-techniques\
/test.csv'
test_data = pd.read_csv(test_file)
X_test = test_data[data_features]
X_test = pd.DataFrame(X_test).fillna(X_test.mean())
print(X_test)
result = data_model.predict(X_test)
print(result)

# %%
# Evaluate model
submission_file = '~/Documents/AIWP/Lab/LabTask_3\
/house-prices-advanced-regression-techniques\
/sample_submission.csv'
submission_data = pd.read_csv(submission_file)
# RMSE && RMSLE
print('RMSE:  ', (mean_absolute_error(
        submission_data['SalePrice'], result)
) ** 0.5)
print('RMSLE: ', (mean_squared_log_error(
        submission_data['SalePrice'], result)
) ** 0.5)

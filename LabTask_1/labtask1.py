import numpy as np
import pandas as pd
import math
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# calculating Root-mean square manually
def MSE(x,y):
    t=0
    k=0
    for i,j in zip(x,y):
        t+= math.pow((i-j),2)
        k+=1
    return t/k
def RMSE(x,y):
    return math.pow(MSE(x,y),0.5)

data = pd.read_csv("Salary_Data.csv")
x = (data.YearsExperience).values.reshape(-1,1)
y = (data.Salary).values.reshape(-1,1)
test = 20 #implies 50%

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = test/100, random_state = 0)
model = LinearRegression().fit(x_train,y_train)
y_pred = model.predict(x_test)

mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print("{0:^38}".format("Results (%d TEST %d TRAIN)"%(test,100-test)))
print("{0:<25} {1:>13.2f}".format("mean-square-error",mse))
print("{0:<25} {1:>13.2f}".format("Root-mean-square-error",rmse))

print("{0:<25} {1:>13.2f}".format("MSE own code: ",MSE(y_test,y_pred)))
print("{0:<25} {1:>13.2f}".format("RMSE own code: ",RMSE(y_test,y_pred)))

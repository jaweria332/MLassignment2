#linear regression

#importing required libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0 , 5.0)

#Reading given Dataset
dataset=pd.read_csv('E:\\python\\dataset.csv')

#collecting x and y
x=dataset['Head Size(cm^3)'].values
y=dataset['Brain Weight(grams)'].values


#calculating mean of X and Y
mean_x=np.mean(x)
mean_y=np.mean(y)

#Capture total number of values in dataset
n = len(x)

#plotting scattered points
plt.scatter(x , y , c='#FF4500' , label = 'Scatter Plot')
plt.title ('Scatter value based on given set of Data')
plt.xlabel('Head Size in cm^3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

plt.show()

#calculating b1 and b2 using formula 
numerator=0
denominator=0
for i in range(n):
    numerator +=(x[i] - mean_x) * (y[i] - mean_y)
    denominator +=(x[i] - mean_x) ** 2
    
b1 = numerator / denominator
b0 = mean_y - (b1 * mean_x)


#plotting values and regression liine
max_x = np.max(x) + 100
min_x = np.min(x) -100

#calculating line values x and y
X = np.linspace(min_x, max_x, 1000)
Y = b0 + b1 * X

#plotting Regression line
plt.plot(X , Y , color='#58b970' , label = 'Regression Line')

#plotting scatter points
plt.scatter(x , y , c='#ef5423' , label = 'Scatter Plot')

#marking x-axis and y-axis
plt.title('Brain weight predicted using Linear Regression')
plt.xlabel('Head Size in cm^3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

#to find out how good our model is using R2 method
ss_t = 0            #for total sum of squares
ss_r = 0            #for total sum of square of residual
for i in range(n):
    y_predict = b0 + b1 * x[i]
    ss_t += (y[i] - mean_y) ** 2
    ss_r += (y[i] - y_predict) ** 2
r2 = 1 - (ss_r/ss_t)
print('R^2 = ' + str(r2))       #R^2 equals to 1 means the prediction is ideal

#To chek accuracy of output using scikit learn Python library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use Rank 1 matrix in scikit learn
x = x.reshape((n, 1))

#creating model
reg = LinearRegression()

#Fitting training data
reg = reg.fit(x, y)

#Y prediction
Y_pred = reg.predict (x)

#calculating RMSE and R2 Score
mse = mean_squared_error(y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(x, y)


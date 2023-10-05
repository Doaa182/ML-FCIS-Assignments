#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#Loading data
data = pd.read_csv('SuperMarketSales.csv')

X=data['Store']
Y=data['Weekly_Sales']

L = 0.0000001  # The learning Rate
epochs = 5*100  # The number of iterations 
m=0  #The slope
c=0  #The y-intercept
n = float(len(X)) # Number of elements in all cols

for i in range(epochs):
    Y_PRED=m*X+c
    dm=-2/n * sum((Y-Y_PRED)*X)
    m=m-L*dm
    dc=-2/n * sum(Y-Y_PRED)
    c=c-L*dc

prediction=m*X+c 


plt.scatter(X, Y)
plt.xlabel('Store', fontsize = 20)
plt.ylabel('Weekly_Sales', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()

mse1=metrics.mean_squared_error(Y, prediction)
print('Mean Square Error of Store ', mse1)

###############################################################################
###############################################################################

X2=data['Date']
Y=data['Weekly_Sales']

#from - to /
data.loc[data['Date'].str.contains('/'),'Date'] = data['Date'].str.replace('/', '-')

#spilit into 3 cols
data[["day", "month", "year"]] = data["Date"].str.split("-", expand = True)

'''
data['day']= data['day'].astype(float)
data['month'] = data['month'].astype(float)
'''
data['year'] = data['year'].astype(float)
#print(data.dtypes)
'''
X2d=data['day']
X2m=data['month']
'''
X2y=data['year']

'''
for i2d in range(epochs):
    Y_PRED2d=m*X2d+c
    dm=-2/n * sum((Y-Y_PRED2d)*X2d)
    m=m-L*dm
    dc=-2/n * sum(Y-Y_PRED2d)
    c=c-L*dc

prediction2d=m*X2d+c


# plt.scatter(X2d, Y)
# plt.xlabel('day', fontsize = 20)
# plt.ylabel('Weekly_Sales', fontsize = 20)
# plt.plot(X2d, prediction2d, color='red', linewidth = 3)
# plt.show()

msed=metrics.mean_squared_error(Y, prediction2d)
print('Mean Square Error of day ', msed)


for i2m in range(epochs):
    Y_PRED2m=m*X2m+c
    dm=-2/n * sum((Y-Y_PRED2m)*X2m)
    m=m-L*dm
    dc=-2/n * sum(Y-Y_PRED2m)
    c=c-L*dc

prediction2m=m*X2m+c


# plt.scatter(X2m, Y)
# plt.xlabel('month', fontsize = 20)
# plt.ylabel('Weekly_Sales', fontsize = 20)
# plt.plot(X2m, prediction2m, color='red', linewidth = 3)
# plt.show()

msem=metrics.mean_squared_error(Y, prediction2m)
print('Mean Square Error of month ',msem )
'''
for i2y in range(epochs):
    Y_PRED2y=m*X2y+c
    dm=-2/n * sum((Y-Y_PRED2y)*X2y)
    m=m-L*dm
    dc=-2/n * sum(Y-Y_PRED2y)
    c=c-L*dc

prediction2y=m*X2y+c


plt.scatter(X2y, Y)
plt.xlabel('year', fontsize = 20)
plt.ylabel('Weekly_Sales', fontsize = 20)
plt.plot(X2y, prediction2y, color='red', linewidth = 3)
plt.show()

msey=metrics.mean_squared_error(Y, prediction2y)
print('Mean Square Error of Year(Date) ', msey)
###############################################################################
###############################################################################
X3=data['Temperature']
Y=data['Weekly_Sales']

for i3 in range(epochs):
    Y_PRED3=m*X3+c
    dm=-2/n * sum((Y-Y_PRED3)*X3)
    m=m-L*dm
    dc=-2/n * sum(Y-Y_PRED3)
    c=c-L*dc

prediction3=m*X3+c

plt.scatter(X3, Y)
plt.xlabel('Temperature', fontsize = 20)
plt.ylabel('Weekly_Sales', fontsize = 20)
plt.plot(X3, prediction3, color='red', linewidth = 3)
plt.show()

mse3=metrics.mean_squared_error(Y, prediction3)
print('Mean Square Error of Temperature ', mse3)



###############################################################################
###############################################################################
X4=data['Fuel_Price']
Y=data['Weekly_Sales']

for i4 in range(epochs):
    Y_PRED4=m*X4+c
    dm=-2/n * sum((Y-Y_PRED4)*X4)
    m=m-L*dm
    dc=-2/n * sum(Y-Y_PRED4)
    c=c-L*dc

prediction4=m*X4+c

plt.scatter(X4, Y)
plt.xlabel('Fuel_Price', fontsize = 20)
plt.ylabel('Weekly_Sales', fontsize = 20)
plt.plot(X4, prediction4, color='red', linewidth = 3)
plt.show()

mse4=metrics.mean_squared_error(Y, prediction4)
print('Mean Square Error of Fuel_Price ',mse4 )

###############################################################################
###############################################################################
X5=data['CPI']
Y=data['Weekly_Sales']

for i5 in range(epochs):
    Y_PRED5=m*X5+c
    dm=-2/n * sum((Y-Y_PRED5)*X5)
    m=m-L*dm
    dc=-2/n * sum(Y-Y_PRED5)
    c=c-L*dc

prediction5=m*X5+c

plt.scatter(X5, Y)
plt.xlabel('CPI', fontsize = 20)
plt.ylabel('Weekly_Sales', fontsize = 20)
plt.plot(X5, prediction5, color='red', linewidth = 3)
plt.show()

mse5= metrics.mean_squared_error(Y, prediction5)
print('Mean Square Error of CPI ',mse5)

###############################################################################
###############################################################################
print("____________________________________________________")
mse1=metrics.mean_squared_error(Y, prediction)
print('Mean Square Error of Store ', mse1)

msey=metrics.mean_squared_error(Y, prediction2y)
print('Mean Square Error of Year(Date) ', msey)

mse3=metrics.mean_squared_error(Y, prediction3)
print('Mean Square Error of Temperature ', mse3)

mse4=metrics.mean_squared_error(Y, prediction4)
print('Mean Square Error of Fuel_Price ',mse4 )

mse5= metrics.mean_squared_error(Y, prediction5)
print('Mean Square Error of CPI ',mse5)
print("____________________________________________________")
print("The Year(Date) Column Has The Minimum Mean Square Error: ")
#print (min(mse1,msed,msem,msey,mse3,mse4,mse5))
print (min(mse1,msey,mse3,mse4,mse5))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import*
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

###############################################################################################
#Loading data
data = pd.read_csv('SuperMarketSales.csv')

#Deal with missing values
#print(data.isna().sum())

###############################################################################################
#from - to /
data.loc[data['Date'].str.contains('/'),'Date'] = data['Date'].str.replace('/', '-')

#spilit into 3 cols
data[["day", "month", "year"]] = data["Date"].str.split("-", expand = True)

#convert to float
data['year'] = data['year'].astype(float)
data['month'] = data['month'].astype(float)
data['day'] = data['day'].astype(float)
#print(data.dtypes)

#drop columns
data=data.drop(['Store','Date'], axis=1)
#data=data.drop(['Store', 'Date','day','month'], axis=1)

market_data=data.iloc[:,:]

###############################################################################################
#Features
X=data.loc[:, data.columns != 'Weekly_Sales']
#Label
Y=data['Weekly_Sales']

###############################################################################################
#Apply Standardization from scratch
def Standardization(X):
    X = np.array(X)
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean( X[:,i])) / np.std( X[:,i])
    return X

X=Standardization(X)

#Apply Built In Standardization 
# scale = StandardScaler()
# newX = scale.fit_transform(X)
# print(newX)

###############################################################################################
#Apply Normalization from scratch
# def Normalization(X,mini,maxi):
#     X = np.array(X)
#     #X=np.zeros((X.shape[0],X.shape[1]))
#     for i in range(X.shape[1]):
#         diff=(max(X[:,i])-min(X[:,i]))
#         stdX=((X[:,i]-min(X[:,i]))/diff)
#         X[:,i]=stdX*(maxi-mini)+mini
#     return X

# X=Normalization(X,0,1)

#Apply Built In Normalization
# scale = MinMaxScaler()
# newX = scale.fit_transform(X)
# print(newX)

###############################################################################################
#correlation bet. features (Feature Selection)
cor = market_data.corr()
(cor['Weekly_Sales']).pop('Weekly_Sales')
new_cor=cor['Weekly_Sales']
#print(new_cor)
strong_corr=(abs(new_cor)).max()
feature1=new_cor.index[abs(new_cor)==strong_corr]

(cor['Weekly_Sales']).pop(feature1[0])
new_cor2=cor['Weekly_Sales']
#print(new_cor2)
strong_corr2=(abs(new_cor2)).max()
feature2=new_cor2.index[abs(new_cor2)==strong_corr2]

top_features = []
top_features.append(feature1[0])
top_features.append(feature2[0])

#Correlation plot
plt.subplots(figsize=(8, 4))
top_corr = market_data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
#top_features = top_features.delete(-1)
X =market_data[top_features]


###############################################################################################
#Built In Polynomial Regression

#Split the data to training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

m1_poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly_m1 = m1_poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_m1 = linear_model.LinearRegression()
#scores1 = cross_val_score(poly_m1, X_train_poly_m1, y_train, scoring='neg_mean_squared_error', cv=5)
scores1 = cross_val_score(poly_m1, X_train_poly_m1, y_train,cv=5)
m1_score = abs(scores1.mean())
poly_m1.fit(X_train_poly_m1, y_train)
print("Model 1 cross validation score is "+ str(m1_score))


m2_poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly_m2 = m2_poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_m2 = linear_model.LinearRegression()
#scores2 = cross_val_score(poly_m2, X_train_poly_m2, y_train, scoring='neg_mean_squared_error', cv=5)
scores2 = cross_val_score(poly_m2, X_train_poly_m2, y_train,cv=5)
m2_score = abs(scores2.mean())
poly_m2.fit(X_train_poly_m2, y_train)
print("Model 2 cross validation score is "+ str(m2_score))

# predicting on test data-set
prediction_builtin1 = poly_m1.predict(m1_poly_features.fit_transform(X_test))
print('Model 1 Test Mean Square Error is', metrics.mean_squared_error(y_test, prediction_builtin1))

# predicting on test data-set
prediction_builtin2 = poly_m2.predict(m2_poly_features.fit_transform(X_test))
print('Model 2 Test Mean Square Error is', metrics.mean_squared_error(y_test, prediction_builtin2))

# # predicting on training data-set
# y_train_predicted = poly_model.predict(X_train_poly)
# ypred=poly_model.predict(poly_features.transform(X_test))

# model_trial(X_train, X_test, y_train, y_test, linear_model.LinearRegression())
# model_trial(X_train, X_test, y_train, y_test, linear_model.Ridge())

###############################################################################################
#Polynomial Regression From Scratch

# The learning Rate
L = 0.0000001 

# The number of iterations to perform gradient descent 
epochs = 1000 
 
# Number of elements in X
m = float(len(X))

theta = (2, 1)
np.zeros(theta)

def hypo(theta, X):
    h = theta*X
    return np.sum(h, axis=1)

def loss(theta, X, Y):
    h = hypo(theta,X)
    return sum((h-Y)**2)/(2*m)

def gradient(theta,X, Y, L,epochs):
    J=[]
    i=0
    while i < epochs:
        h= hypo(theta,X)
        for k in range(0, len(X.columns)):
            theta[k] = theta[k] - L*(sum((h-Y)* X.iloc[:, k])/m)
        j = loss(theta,X, Y)
        J.append(j)
        i =i+ 1
    return J, theta


prediction_scratch = hypo(theta,X)

print('Mean Square Error (From Scratch) is ', metrics.mean_squared_error(np.asarray(Y), prediction_scratch))

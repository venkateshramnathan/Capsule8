# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:32:22 2017

@author: Venkatesh
"""
import pandas
import numpy as np
#from math import sqrt 
from pandas import Series
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
#from sklearn.utils import check_array
from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt

df = pandas.read_csv("C:/Users/user/Desktop/Drchen/Cap8/capsule8.csv")
df1 = pandas.read_excel("C:/Users/user/Desktop/Drchen/Cap8/cap8.xlsx")
df2 = pandas.read_excel("C:/Users/user/Desktop/Drchen/Cap8/tcp_life.xlsx")

x = Series.from_array(df['ts'])
y = Series.from_array(df['depth'])
#binwidth = 200

# 1.time-series & histogram for ts
x.hist()
x.plot()

#time-series & histogram for depth
y.hist()
plt.plot(x,y)

plot_1 = plt.plot(df['ts'])
plot_2 = plt.plot(df['ts'],df['depth'])

#time - series & histogram for rate_1 & rate_5
df1['rate_1']=np.nan
df1['rate_5']=np.nan
for i in range(0, len(df1['rates'])):
    for j in eval(df1['rates'][i]):
        if j == '1':            
            df1['rate_1'][i] = (eval(df1['rates'][i])[j])
        else:
            df1['rate_5'][i] = (eval(df1['rates'][i])[j])

plot_3 = plt.plot(df1['ts'],df1['rate_1'])
plot_4 = plt.plot(df1['ts'],df1['rate_5'])

df1['rate_1'].hist()
df1['rate_5'].hist()

# 3. time-series & histogram for rx and tx
plot_5 = plt.plot(df2['ts'],df2['rx'])
plot_6 = plt.plot(df2['ts'],df2['tx']) 

df2['rx'].hist()
df2['tx'].hist()

#time-series & histogram for dur
plot_7 = plt.plot(df2['ts'],df2['dur'])
df2['dur'].hist()
#df2['dur'].hist(bins=np.arange(min(df2['dur']), max(df2['dur']) + binwidth, binwidth))

#time-series & histogram for lport & rport
plot_8 = plt.plot(df2['ts'],df2['lport'])
plot_9 = plt.plot(df2['ts'],df2['rport'])

df2['lport'].hist()
df2['rport'].hist()

Series.from_array(df2['dur']).plot(kind='kde')
#plot_10 = plt.vlines(df2['dur'].mean(),ymin=0,ymax=0.0025,linewidth=2.0)
#plot_11 = plt.vlines(df2['dur'].median(),ymin=0,ymax=0.0025,linewidth=2.0,color="red")

df3 = pandas.read_csv("C:/Users/user/Desktop/Drchen/Cap8/dur1.csv")
df4 = pandas.read_csv("C:/Users/user/Desktop/Drchen/Cap8/dur2.csv")
df5 = pandas.read_csv("C:/Users/user/Desktop/Drchen/Cap8/dur3.csv")

#df3['dur1'].hist(bins=np.arange(min(df3['dur1']), max(df3['dur1']) + binwidth, binwidth))
df3['dur1'].hist()
df4['dur2'].hist()
df5['dur3'].hist()


df2['dur'].describe()

upper = 226.797500
lower = 85.512500
IQ = upper - lower

lower_inner_fence = lower - 1.5*(IQ)
upper_inner_fence = upper + 1.5*(IQ)
lower_outer_fence = lower - 3.0*(IQ)
upper_outer_fence = upper + 3.0*(IQ)


df2['dur'].plot.box(vert = False)
plt.show()


#Data Modelling 
df2['443'] = np.nan #extracting values of histoports 443 & 80
df2['80'] = np.nan
for i in range(0, len(df2['histoports'])):
    x = eval(df2['histoports'][i])
    x = x[1]
    df2['443'][i] = x['443']
    df2['80'][i] = x['80']

df2.to_csv('model.csv', encoding='utf-8')    

model_dataset = pandas.read_csv("C:/Users/user/Desktop/Drchen/Cap8/model.csv")
model_dataset['rport'] = model_dataset['rport'].astype(str)
model_dataset['lport'] = model_dataset['lport'].astype(str)
model_dataset['predquality'] = model_dataset['predquality'].astype(str)

random_forest_model = RandomForestRegressor(random_state=0)
X = model_dataset.iloc[:,1:9]
Y = model_dataset.iloc[:,0]

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3)


param_grid = {'n_estimators': [17, 18, 19, 20, 22, 23, 25], 
              'max_depth': [10, 11, 12, 13, 14, 15], 
              }

grid_clf = GridSearchCV(random_forest_model, param_grid, cv = 10)
grid_clf1 = GridSearchCV(random_forest_model, param_grid, cv = 5)
grid_clf2 = GridSearchCV(random_forest_model, param_grid, cv = 3)

def mape(y_pred,y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
 
grid_clf.fit(X_train, Y_train) #fitting training on 10 fold gridCV
grid_clf.best_estimator_
grid_clf.best_params_
grid_clf.best_score_

grid_clf.score(X_train, Y_train)
grid_clf.score(X_test, Y_test)


temp2 = grid_clf.predict(X_train) #predicting in-sample for 10 fold gridCV 
temp = grid_clf.predict(X_test) #predicting out-sample for 10 fold gridCV
mape(Y_train.values, temp2)
mape(Y_test.values, temp)

#insample plot
fig, ax = plt.subplots()
ax.scatter(Y_train, temp2, edgecolors=(0, 0, 0))
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#outsample plot
fig, ax = plt.subplots()
ax.scatter(Y_test, temp, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()


grid_clf1.fit(X_train, Y_train) #fitting training on 5 fold gridCV
grid_clf1.best_estimator_
grid_clf1.best_params_
grid_clf1.best_score_

grid_clf1.score(X_train, Y_train)
grid_clf1.score(X_test, Y_test)

temp3 = grid_clf1.predict(X_train)
temp1 =grid_clf1.predict(X_test)
mape(Y_train.values, temp3)
mape(Y_test.values, temp1)

#insample plot
fig, ax = plt.subplots()
ax.scatter(Y_train, temp3, edgecolors=(0, 0, 0))
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#outsample plot
fig, ax = plt.subplots()
ax.scatter(Y_test, temp1, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()



grid_clf2.fit(X_train, Y_train) #fitting training on 3 fold gridCV
grid_clf2.best_estimator_
grid_clf2.best_params_
grid_clf2.best_score_

grid_clf2.score(X_train, Y_train)
grid_clf2.score(X_test, Y_test)

temp5 = grid_clf2.predict(X_train)
temp4 =grid_clf2.predict(X_test)
mape(Y_train.values, temp5)
mape(Y_test.values, temp4)

#insample plot
fig, ax = plt.subplots()
ax.scatter(Y_train, temp5, edgecolors=(0, 0, 0))
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#outsample plot
fig, ax = plt.subplots()
ax.scatter(Y_test, temp4, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

''' some other stuffs that i tried
pca = PCA(n_components=2)
pca1 = PCA(n_components=3)
pca2 = PCA(n_components=4)

standardized_X = preprocessing.scale(X)
pca.fit(standardized_X)
pca1.fit(standardized_X)
pca2.fit(standardized_X)


pca.explained_variance_ratio_
pca1.explained_variance_ratio_
pca2.explained_variance_ratio_

X1 = pca.fit_transform(standardized_X)
X2 = pca1.fit_transform(standardized_X)
X3 = pca2.fit_transform(standardized_X)

X1_train, X1_test = train_test_split( X1, test_size = 0.1)
X2_train, X2_test = train_test_split( X2, test_size = 0.1)
X3_train, X3_test = train_test_split( X3, test_size = 0.1)

pandas.DataFrame(X1)
pandas.DataFrame(X2)
pandas.DataFrame(X3)


reg1 = random_forest_model.fit(X1_train, Y_train)
model1_out = reg1.predict(X1_test)
reg1.score(X1_train, Y_train)
reg1.score(X1_test,Y_test) 

reg2 = random_forest_model.fit(X_train.iloc[:,[2,3,6,7,1,4,5]], Y_train)
model2_out = reg2.predict(X_test.iloc[:,[2,3,6,7,1,4,5]])
reg2.score(X_train.iloc[:,[2,3,6,7,1,4,5]], Y_train) 
reg2.score(X_test.iloc[:,[2,3,6,7,1,4,5]],Y_test)


reg3 = random_forest_model.fit(X_train.iloc[:,[0,2,3,6,7,1,4,5]], Y_train)
model3_out = reg3.predict(X_test.iloc[:,[0,2,3,6,7,1,4,5]])
reg3.score(X_train.iloc[:,[0,2,3,6,7,1,4,5]], Y_train) 
reg3.score(X_test.iloc[:,[0,2,3,6,7,1,4,5]],Y_test)

reg4 = random_forest_model.fit(X2_train, Y_train)
model4_out = reg1.predict(X2_test)
reg4.score(X2_train, Y_train)
reg4.score(X2_test,Y_test)

reg5 = random_forest_model.fit(X3_train, Y_train)
model4_out = reg1.predict(X3_test)
reg4.score(X3_train, Y_train)
reg4.score(X3_test,Y_test)

reg2 = random_forest_model.fit(X_train.iloc[:,[2,3,1,4,5]], Y_train)
model2_out = reg2.predict(X_test.iloc[:,[2,3,1,4,5]])
reg2.score(X_train.iloc[:,[2,3,1,4,5]], Y_train) 
reg2.score(X_test.iloc[:,[2,3,1,4,5]],Y_test)
#explained_variance_score(model1_out, Y_test)
#explained_variance_score(model2_out, Y_test) 
#z = random_forest_model.predict(model_dataset.iloc[841:943,[1,2,4]])
'''

#fitting random forest only with ts
reshaped_features = X_train.iloc[:,3].values.reshape(-1 ,1)
reshaped_features_test = X_test.iloc[:,3].values.reshape(-1 ,1)

random_forest_model.fit(reshaped_features, Y_train)
random_forest_model.score(reshaped_features, Y_train)
random_forest_model.score(reshaped_features_test, Y_test)


train_predict = random_forest_model.predict(reshaped_features)  
test_predict = random_forest_model.predict(reshaped_features_test) 
mape(Y_train.values, train_predict)
mape(Y_test.values, test_predict)

#insample plot
fig, ax = plt.subplots()
ax.scatter(Y_train, train_predict, edgecolors=(0, 0, 0))
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#outsample plot
fig, ax = plt.subplots()
ax.scatter(Y_test, test_predict, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()




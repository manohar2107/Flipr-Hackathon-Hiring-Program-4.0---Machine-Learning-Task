#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
ts_data= pd.read_excel('Train_dataset.xlsx',sheet_name='Diuresis_TS')
test_dataset1=pd.read_excel("Test_dataset.xlsx")
x=ts_data.iloc[:,2:7].values
y=ts_data.iloc[:,7].values
x_t1=ts_data.iloc[:,3:8].values
x_t2=ts_data.iloc[:,1:2].values
x_t3=test_dataset1.iloc[:,16:17].values
#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
#Calculating Diuresis Of 27th March
y_pred2 =regressor.predict(x_t1)

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_t2,y_pred2)
#Calculating Diuresis Of 27th March
y_pred1 =regressor.predict(x_t3)

#Dataset
train_dataset=pd.read_excel("Train_dataset.xlsx")
x_train=train_dataset.iloc[:,[0,3,5,6,7,8,9,11,13,15,16,17,18,19,21]]
y_train=train_dataset.iloc[:,27].values
test_dataset=pd.read_excel("Test_dataset2.xlsx")
x_test=test_dataset.iloc[:,[0,3,5,6,7,8,9,11,13,15,16,17,18,19,21]]
#Missing Values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean',verbose=0)
imputer = imputer.fit(x_train.iloc[: ,[3,10,11,12,13,14]])
x_train.iloc[: ,[3,10,11,12,13,14]]= imputer.transform(x_train.iloc[: ,[3,10,11,12,13,14]])
#Missing value in Occupation
p = x_train.Occupation.value_counts(normalize=True)  
m = x_train.Occupation.isnull()
np.random.seed(42)
rand_fill = np.random.choice(p.index, size=m.sum(), p=p)
x_train.loc[m, 'Occupation'] = rand_fill
#Missing value in Mode_transport
p = x_train.Mode_transport.value_counts(normalize=True)  
m = x_train.Mode_transport.isnull()
np.random.seed(42)
rand_fill = np.random.choice(p.index, size=m.sum(), p=p)
x_train.loc[m, 'Mode_transport'] = rand_fill
#Missing value in comorbidity
p = x_train.comorbidity.value_counts(normalize=True)  
m = x_train.comorbidity.isnull()
np.random.seed(42)
rand_fill = np.random.choice(p.index, size=m.sum(), p=p)
x_train.loc[m, 'comorbidity'] = rand_fill
#Missing value in cardiological_pressure
p = x_train.cardiological_pressure.value_counts(normalize=True)  
m = x_train.cardiological_pressure.isnull()
np.random.seed(42)
rand_fill = np.random.choice(p.index, size=m.sum(), p=p)
x_train.loc[m, 'cardiological_pressure'] = rand_fill
#categorical data(training)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [1,2,4,5,7,9])],remainder='passthrough')
# transform training data
x_train = transformer.fit_transform(x_train)
#categorical data(test)
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [1,2,4,5,7,9])],remainder='passthrough')
# transform testing data
x_test = transformer.fit_transform(x_test)

#Building a optimal Model
import statsmodels.api as sm
x_train=np.append(arr =np.ones((10714,1)).astype(int) ,values=x_train,axis=1)
x_test=np.append(arr =np.ones((14498,1)).astype(int) ,values=x_test,axis=1)
x_opt=x_train[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]
regressor_OLS=sm.OLS(endog=y_train,exog=x_opt).fit()
regressor_OLS.summary()

#Fitting Multiple Regression in Training Set
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)

#Prediction
y_pred=regressor.predict(x_test)



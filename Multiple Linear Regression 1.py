import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df_house = pd.read_csv("C:/Users/USER/Desktop/DataSet/Housing_Price.csv")
print(df_house.head(5))
print("Missing Values")
print(df_house.isnull().sum())

df_house.replace(to_replace='yes',value=1,inplace=True)
df_house.replace(to_replace='no',value=0,inplace=True)
df_house.replace(to_replace='unfurnished',value=0,inplace=True)
df_house.replace(to_replace='semi-furnished',value=1,inplace=True)
df_house.replace(to_replace='furnished',value=2,inplace=True)
print(df_house.head())


x=df_house.drop('price',axis=1)
y=df_house['price']

X_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)

"""y_train_2d= np.array(y_train).reshape(-1,1)
y_test= np.array(y_test).reshape(-1,1)
print(y_train.shape)
"""
model=LinearRegression()
model.fit(X_train,y_train)

print("Intercept (c):", model.intercept_)

'''for item in list (zip(x.columns.values, model.coef_)):
    print(f"{item[0]}".ljust(15, " " ),f"{item[1]:.6f}")'''

f=x.columns
c=model.coef_
d=pd.DataFrame({'features':f,'reg coef':c})
print(d)


y_pred = model.predict(x_test)

mse =mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print ("mean_squared_error :",mse)
print ("r2_score :",r2)
print ("mean_absolute_error :",mae)
print ("root_mean_squared_error :",rmse)


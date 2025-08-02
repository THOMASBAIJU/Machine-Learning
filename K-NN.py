import pandas as pd
import numpy as np
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error,mean_absolute_error

df_comp = pd.read_csv("C:/Users/USER/Desktop/DataSet/Company_data.csv")
print(df_comp.head(5))

print("Missing Values")
print(df_comp.isnull().sum())

x_features=df_comp.drop("Sales" , axis= 1)
y_features=df_comp['Sales']

ob = StandardScaler()
scaledFeatures = ob.fit_transform(x_features)
x_scaled = pd.DataFrame(scaledFeatures)
x_scaled.columns = x_features.columns
print(x_scaled)

X_train,X_test,y_train,y_test =train_test_split(x_features,y_features,train_size=0.7,random_state=42)

model = KNeighborsRegressor(n_neighbors=2)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
y_pred=model.predict(X_test)

mse =mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print("Mean squared error :",mse)
print("R2 Score :",r2)
print("Mean Absolute Error:",mae)
print("Rooted Mean Squared Error",rmse)


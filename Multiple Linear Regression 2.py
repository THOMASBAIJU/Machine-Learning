#implement multiple linear regression using company_dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

df = pd.read_csv("C:/Users/USER/Desktop/DataSet/Company_data.csv")
print(df.head(5))

print("Missing Values")
print(df.isnull().sum())

x=df.drop("Sales" , axis= 1)
y=df['Sales']

X_train,X_test,y_train,y_test =train_test_split(x,y,train_size=0.7,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

intercept = model.intercept_
print("Intercept (c):", intercept)

f=x.columns
c=model.coef_
d=pd.DataFrame({'features':f,'reg coef':c})
print(d)

y_pred=model.predict(X_test)

mse =mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)

print("Mean squared error :",mse)
print("R2 Score :",r2)
print("Mean Absolute Error:",mae)
print("Rooted Mean Squared Error",)

def spend_predict(tv_val,radio_val,news):
    input_df = pd.DataFrame([[tv_val, radio_val, news]], columns=['TV', 'Radio', 'Newspaper'])
    prediction = model.predict(input_df)
    print("Predicted Sales:", prediction)

tv_val1 = int(input("TV: "))
radio_val1 = int(input("Radio: "))
news1 = int(input("NEWS: "))
spend_predict(tv_val1, radio_val1, news1)
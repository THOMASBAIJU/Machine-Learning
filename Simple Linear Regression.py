2import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_ins = pd.read_csv("D:/Dataset/insurance_dataset.csv")
print(df_ins.head())
print("missing values")
print(df_ins.isnull().sum())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

sns.regplot(x="age",y="charges",data=df_ins,line_kws={"color":"red"})
plt.title("Age vs Charge")
plt.xlabel("age")
plt.ylabel("charges")

plt.show()


X = df_ins['age'].values.reshape(-1, 1)
y = df_ins['charges'].values.reshape(-1, 1)
print("Shape of feature (X) array:", X.shape)
print("Shape of target (y) array:", y.shape)

X_train,X_test,y_train,y_test = train_test_split(df_ins[["age"]],df_ins["charges"],test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

def charges_prediction(age_value):
    return model.predict(pd.DataFrame([[age_value]], columns=["age"]))

print("Intercept (c):", model.intercept_)
print("Coefficient (m):", model.coef_[0])

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae= mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print("mean_squared_error : ",mse)
print("R-squared : ",r2)
print("Mean Absolute Error  : ",mae)
print("Root mean Squared Error :", rmse)

n =int(input("Age :"))
print(charges_prediction(n))
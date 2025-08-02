import pandas as pd
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

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

X_train,X_test,y_train,y_test =train_test_split(x_scaled,y_features,train_size=0.8,random_state=89)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train,y_train)

y_predtrain=model.predict(X_train)
y_predtest=model.predict(X_test)

print("Accuracy test:", r2_score(y_test,y_predtest))
print("Accuracy train:", r2_score(y_train,y_predtrain))

def spend_predict(tv,radio,news):
    input_df = pd.DataFrame([[tv,radio,news]])
    prediction = model.predict(input_df)
    print("Predicted Sales:", prediction)

tv_val = float(input("TV: "))
radio_val = float(input("Radio: "))
news_val= float(input("NEWS: "))
spend_predict(tv_val, radio_val, news_val)

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("C:/Users/USER/Desktop/DataSet/Social_Network_Ads.csv")
print(df.head())

print("Missing Values")
print(df.isnull().sum())

x_features = df.drop(['User ID','Purchased'],axis=1)
y_feather = df['Purchased']

X_train,X_test,y_train,y_test = train_test_split(x_features,y_feather, test_size=0.2, random_state= 42)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train,y_train)

y_predtrain=model.predict(X_train)
y_predtest=model.predict(X_test)

print("Accuracy test:",accuracy_score(y_test,y_predtrain))
print("Accuracy train:",accuracy_score(y_test,y_predtrain))

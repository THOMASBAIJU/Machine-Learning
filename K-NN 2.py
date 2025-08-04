import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:/Users/USER/Desktop/DataSet/Social_Network_Ads.csv")
print(df.head())
print("Missing Values")
print(df.isnull().sum())

x_features = df.drop(['User ID','Purchased'], axis=1)
y_feature = df['Purchased']

print("x_features")
print(x_features)
print('y_features')
print(y_feature)

x_features = pd.get_dummies(x_features, columns=['Gender'])
x_features.info()


X_train, X_test, y_train, y_test = train_test_split(x_features, y_feature, test_size=0.2, random_state=42)

print("Shape of feature (X) array:",x_features.shape)
print("Shape of feature (Y) array:",y_feature.shape)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print("Accuracy train:", accuracy_score(y_train, y_pred_train))
print("Accuracy test:", accuracy_score(y_test, y_pred_test))

print("classification_report of Test:")
print(classification_report(y_test,y_pred_test))
print("classification_report of Train")
print(classification_report(y_train,y_pred_train))
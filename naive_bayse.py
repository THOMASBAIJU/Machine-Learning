#import Modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB

#load iris_dataset and do train_test_split
df=pd.read_csv('https://raw.githubusercontent.com/THOMASBAIJU/IRIS_DATASET/refs/heads/main/IRIS.csv')
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#implement naive bayes
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#predict values for test data
y_pred=classifier.predict(X_test)

#Display accuracy score and display confusion matrix and classification report
pred=classifier.predict(X_test)
print("Accuracy : ",accuracy_score(y_test,y_pred))
print("Confusion Matrix : \n",confusion_matrix(y_test,y_pred))
print("Classification Report : \n",classification_report(y_test,pred))
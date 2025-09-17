# Import Modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('C:/Users/USER/Desktop/DataSet/Breast_Cancer.csv')
print(df.head())

# Display basic statistics and check for missing values
print(df.describe())
print(df.isnull().sum())

# Drop unnecessary columns
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Split features and target
x = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Perform Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Construct Decision Tree classifier
dt_model = DecisionTreeClassifier(criterion="entropy", min_samples_split=2)
dt_model.fit(x_train, y_train)

# Predict on test data
y_pred = dt_model.predict(x_test)
print("Predicted labels:", y_pred)

# Display accuracy
print("Accuracy on test data:", accuracy_score(y_test, y_pred))

# Print Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot Decision Tree
plt.figure(figsize=(15, 15))
tree.plot_tree(dt_model, filled=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.show()

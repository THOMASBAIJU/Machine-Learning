# Import Modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
from graphviz import Source
from PIL import Image
import io

# Load iris_dataset and remove missing values
df = pd.read_csv('https://raw.githubusercontent.com/THOMASBAIJU/IRIS_DATASET/refs/heads/main/IRIS.csv')
df = df.dropna()
print(df.head())
print(df.shape)

# Split features and target
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature and class names
fname = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
tname = y.unique().tolist()

# Implement Decision Tree
model = DecisionTreeClassifier(criterion="entropy", min_samples_split=15)
model.fit(X_train, y_train)

# Export tree to DOT format and visualize
dot_data = export_graphviz(model, out_file=None,
                           feature_names=fname,
                           class_names=tname,
                           filled=True, rounded=True)
graph = Source(dot_data)

# Save the tree as PNG and display it
graph.render("Iris_Decision_Tree", view=True, format='png')
print("\nDecision Tree visualization saved as Iris_Decision_Tree.png")

# Display the image directly using PIL
img = Image.open(io.BytesIO(graph.pipe(format="png")))
img.show()

# Predict values for test data
y_pred = model.predict(X_test)

# Display accuracy score and confusion matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
            xticklabels=tname, yticklabels=tname)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot the Decision Tree using matplotlib
plt.figure(figsize=(15, 15))
tree.plot_tree(model, filled=True, fontsize=10, feature_names=fname, class_names=tname)
plt.title("Decision Tree Structure")
plt.show()

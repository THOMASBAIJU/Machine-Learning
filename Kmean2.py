import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
x= iris.data
y=iris.target
features = iris.feature_names

print("X features\n",x)
print("Target\n",y)
print("Features :",features)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=10, n_init=10)
kmeans.fit(x)


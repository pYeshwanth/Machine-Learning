import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X)

print("Cluster Labels:", kmeans.labels_[:10])
print("Cluster Centers:\n", kmeans.cluster_centers_)
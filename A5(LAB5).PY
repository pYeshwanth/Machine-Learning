import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X)

labels = kmeans.labels_

print("Silhouette Score:", silhouette_score(X, labels))
print("CH Score:", calinski_harabasz_score(X, labels))
print("DB Index:", davies_bouldin_score(X, labels))
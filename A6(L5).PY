import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values

k_values = range(2, 10)
scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)
    scores.append(silhouette_score(X, km.labels_))

plt.plot(k_values, scores)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()
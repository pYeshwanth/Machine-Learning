import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values

agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(X)

db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(X)

print("Agglomerative Silhouette:", silhouette_score(X, agg_labels))

valid = db_labels != -1
if len(set(db_labels[valid])) > 1:
    print("DBSCAN Silhouette:", silhouette_score(X[valid], db_labels[valid]))
else:
    print("DBSCAN could not form valid clusters")
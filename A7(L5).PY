import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values

distortions = []

for k in range(2, 20):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(2, 20), distortions)
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.show()
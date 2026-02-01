import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
X = numeric_data.iloc[:, :-1].values

def minkowski_distance(A, B, p):
    return sum(abs(a-b)**p for a,b in zip(A,B))**(1/p)

v1 = X[0]
v2 = X[1]

distances = [minkowski_distance(v1, v2, p) for p in range(1,11)]

plt.plot(range(1,11), distances)
plt.xlabel("p")
plt.ylabel("Distance")
plt.show()

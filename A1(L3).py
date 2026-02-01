import numpy as np
import pandas as pd
data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])

X = numeric_data.iloc[:, :-1].values


def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

def euclidean_norm(A):
    return sum(a * a for a in A) ** 0.5

A = X[0]
B = X[1]

print("Custom Dot:", dot_product(A, B))
print("NumPy Dot:", np.dot(A, B))

print("Custom Norm:", euclidean_norm(A))
print("NumPy Norm:", np.linalg.norm(A))

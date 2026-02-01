import numpy as np
import pandas as pd

data = pd.read_csv("features_with_labels(in).csv")
numeric_data = data.select_dtypes(include=[np.number])

X = numeric_data.iloc[:, :-1].values
y = numeric_data.iloc[:, -1].values

classes = np.unique(y)[:2]

c1 = X[y == classes[0]]
c2 = X[y == classes[1]]

mean1 = np.mean(c1, axis=0)
mean2 = np.mean(c2, axis=0)

spread1 = np.std(c1, axis=0)
spread2 = np.std(c2, axis=0)

dist = np.linalg.norm(mean1 - mean2)

print("Spread Class1:", spread1)
print("Spread Class2:", spread2)
print("Interclass Distance:", dist)

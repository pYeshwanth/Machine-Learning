import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
X = numeric_data.iloc[:, :-1].values

v1 = X[0]
v2 = X[1]

print("SciPy Minkowski:", minkowski(v1, v2, 3))

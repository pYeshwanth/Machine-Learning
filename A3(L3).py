import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
X = numeric_data.iloc[:, :-1].values

feature = X[:, 0]

print("Mean:", np.mean(feature))
print("Variance:", np.var(feature))

plt.hist(feature, bins=10)
plt.title("Feature Histogram")
plt.show()

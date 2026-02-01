import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
X = numeric_data.iloc[:, :-1].values
y = numeric_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Train Size:", len(X_train))
print("Test Size:", len(X_test))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values
y = numeric_data.iloc[:, -1].values

y = (y > np.median(y)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

def confusion_values(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score_custom(p, r):
    return 2 * p * r / (p + r) if (p + r) != 0 else 0

tp, tn, fp, fn = confusion_values(y_test, predictions)

acc = accuracy(tp, tn, fp, fn)
prec = precision(tp, fp)
rec = recall(tp, fn)
f1 = f1_score_custom(prec, rec)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

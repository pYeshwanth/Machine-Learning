import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx")

df = df.select_dtypes(include=["int64","float64"]).dropna()

corr = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
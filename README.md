```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display dataset information
print(df.head())
print(df.describe())
df.info()

# 1. Histogram
df.hist(figsize=(10, 6))
plt.show()

# 2. Box Plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)',
                y='petal length (cm)', data=df)
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

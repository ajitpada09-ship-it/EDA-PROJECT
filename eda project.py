import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("titanic.csv")

# Basic Information
print("First 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# Missing values
print("\nMissing Values")
print(df.isnull().sum())

# Histogram
df["Age"].hist()
plt.title("Age Distribution")
plt.savefig("histogram.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))
plt.imshow(df.corr(numeric_only=True), cmap="coolwarm")
plt.colorbar()
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Boxplot
plt.figure(figsize=(5,5))
plt.boxplot(df["Fare"].dropna())
plt.title("Fare Boxplot")
plt.savefig("boxplot.png")
plt.show()
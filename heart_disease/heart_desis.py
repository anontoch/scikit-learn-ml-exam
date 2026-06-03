import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("heart.csv")
print(df.head())

# 2. Count disease vs non-disease
# Assuming 'target' column: 1 = disease, 0 = no disease
disease_count = df["target"].value_counts()
print("\nDisease (1) vs No Disease (0):")
print(disease_count)

# 3. Scatter plot: Age vs Disease
plt.figure(figsize=(6,4))
plt.scatter(df["age"], df["target"])
plt.xlabel("Age")
plt.ylabel("Disease (1=yes, 0=no)")
plt.title("Age vs Disease")
plt.grid(True)
plt.show()

# 4. Scatter plot: Max Heart Rate vs Disease
plt.figure(figsize=(6,4))
plt.scatter(df["thalach"], df["target"])
plt.xlabel("Maximum Heart Rate (thalach)")
plt.ylabel("Disease (1=yes, 0=no)")
plt.title("Max Heart Rate vs Disease")
plt.grid(True)
plt.show()

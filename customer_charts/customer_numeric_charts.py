import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv("customers-100.csv")

# Detect numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
print("Numeric columns:", list(numeric_cols))

# If no numeric columns found, stop (your CSV is mostly strings)
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns available for plotting.")

# ============================
# 1. LINE CHART
# ============================
for col in numeric_cols:
    plt.figure(figsize=(10,5))
    plt.plot(df[col])
    plt.title(f"Line Chart of {col}")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

# ============================
# 2. BAR CHART (Top 10 values)
# ============================
for col in numeric_cols:
    plt.figure(figsize=(10,5))
    df[col].head(10).plot(kind="bar")
    plt.title(f"Bar Chart of First 10 Values - {col}")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

# ============================
# 3. HISTOGRAM
# ============================
for col in numeric_cols:
    plt.figure(figsize=(10,5))
    plt.hist(df[col], bins=15)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# ============================
# 4. SCATTER PLOT
# (Pair numeric columns)
# ============================
if len(numeric_cols) >= 2:
    for i in range(len(numeric_cols) - 1):
        plt.figure(figsize=(7,5))
        plt.scatter(df[numeric_cols[i]], df[numeric_cols[i+1]])
        plt.title(f"Scatter: {numeric_cols[i]} vs {numeric_cols[i+1]}")
        plt.xlabel(numeric_cols[i])
        plt.ylabel(numeric_cols[i+1])
        plt.grid(True)
        plt.show()

# ============================
# 5. PIE CHART (categorical example)
# ============================
if "Country" in df.columns:
    plt.figure(figsize=(8,8))
    df["Country"].value_counts().head(5).plot(kind="pie", autopct='%1.1f%%')
    plt.title("Top 5 Countries Distribution")
    plt.ylabel("")
    plt.show()

# ============================
# 6. BOX PLOT
# ============================
plt.figure(figsize=(10,6))
df[numeric_cols].plot(kind="box")
plt.title("Boxplot of Numeric Columns")
plt.grid(True)
plt.show()

# ============================
# 7. KDE PLOT (distribution curve)
# ============================
for col in numeric_cols:
    plt.figure(figsize=(8,5))
    sns.kdeplot(df[col], fill=True)
    plt.title(f"KDE Density - {col}")
    plt.xlabel(col)
    plt.grid(True)
    plt.show()

# ============================
# 8. HEATMAP (correlation matrix)
# ============================
plt.figure(figsize=(8,6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

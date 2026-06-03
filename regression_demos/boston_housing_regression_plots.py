# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import boston_housing
from sklearn.linear_model import LinearRegression

# --- Load Dataset ---
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print("Training data shape:", X_train.shape)
print("First training sample features:", X_train[0])
print("Target price for the first sample:", y_train[0])

# --- Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Display Results ---
print("Predicted housing prices:", y_pred)
print("Actual housing prices:", y_test)
print("Model R² score:", model.score(X_test, y_test))

# --- Visualization 1: Predicted vs Actual Prices ---
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices ($1000s)")
plt.ylabel("Predicted Prices ($1000s)")
plt.title("Predicted vs Actual Housing Prices")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)  # perfect prediction line
plt.show()

# --- Visualization 2: Prediction Error Distribution ---
errors = y_pred - y_test
plt.figure(figsize=(8,6))
sns.histplot(errors, bins=25, kde=True)
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error ($1000s)")
plt.ylabel("Count")
plt.show()

# --- Visualization 3: Feature Correlation Heatmap ---
# Boston dataset has 13 features; get their names
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
# Combine features and target into a DataFrame for analysis
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df["PRICE"] = y_train

plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation with House Price")
plt.show()

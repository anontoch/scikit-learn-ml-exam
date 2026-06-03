import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------------------
# Load CSV
# ------------------------------

df = pd.read_csv("train.csv")

# ------------------------------
# Separate features & label
# ------------------------------

X_df = df.iloc[:, :-1]
y = df.iloc[:, -1].values

# ------------------------------
# Handle categorical features
# One-Hot Encoding (manual, no sklearn)
# ------------------------------

X_df = pd.get_dummies(X_df)

# ------------------------------
# Convert to NumPy
# ------------------------------

X = X_df.values.astype(np.float32)

# ------------------------------
# Handle missing values
# ------------------------------

X = np.nan_to_num(X)

# ------------------------------
# Normalize (DNN requirement)
# ------------------------------

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X = (X - X_mean) / X_std

# ------------------------------
# Split data (70/15/15)
# ------------------------------

np.random.seed(42)
indices = np.random.permutation(len(X))

train_end = int(0.7 * len(X))
val_end = int(0.85 * len(X))

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# ------------------------------
# Task 2.1: Build DNN
# ------------------------------

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------
# Task 2.2: Train & Validate
# ------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ------------------------------
# Task 2.3: Evaluate
# ------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

y_pred = (model.predict(X_test) > 0.5).astype(int)
manual_acc = (y_pred.flatten() == y_test).mean()
print("Manual Test Accuracy:", manual_acc)
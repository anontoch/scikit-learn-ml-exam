import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# ------------------------------
# 1. Generate synthetic dataset
# ------------------------------

# Example: 1000 samples, 20 features, binary classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ------------------------------
# 2. Function to build model
# ------------------------------

def build_model(input_dim, optimizer):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ------------------------------
# 3. Different Optimizers
# ------------------------------

optimizers = {
    "SGD": SGD(learning_rate=0.01),
    "RMSprop": RMSprop(learning_rate=0.001),
    "Adam": Adam(learning_rate=0.001)
}

results = {}

for name, opt in optimizers.items():
    print(f"\nTraining with optimizer: {name}")

    model = build_model(X_train.shape[1], opt)

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    results[name] = test_acc
    print(f"{name} Test Accuracy: {test_acc:.4f}")

# ------------------------------
# 4. Final comparison
# ------------------------------

print("\nOptimizer Comparison Results:")
for opt, acc in results.items():
    print(f"{opt}: {acc:.4f}")
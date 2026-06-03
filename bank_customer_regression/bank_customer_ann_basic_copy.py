import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read data
df_bank = pd.read_csv("dataset/BankCustomer.csv")
print("First 5 rows of data:")
print(df_bank.head())
print("\nData Info:")
print(df_bank.info())

# Display data distribution
features = ['City', 'Gender', 'Age', 'Tenure',
            'ProductsNo', 'HasCard', 'ActiveMember', 'Exited']

# Create visualization
fig, axes = plt.subplots(4, 2, figsize=(15, 15))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

for i, feature in enumerate(features):
    ax = axes[i//2, i%2]
    sns.countplot(x=feature, data=df_bank, ax=ax)
    ax.set_title(f"No. of customers - {feature}")
    if feature in ['City', 'Gender']:
        ax.tick_params(axis='x', rotation=45)

plt.show()

# Data preprocessing
# Encode Gender column
df_bank['Gender'] = df_bank['Gender'].map({'Female': 0, 'Male': 1})
print("\nGender unique values after encoding:", df_bank['Gender'].unique())

# Convert categorical City to dummy variables
d_city = pd.get_dummies(df_bank['City'], prefix="City")
df_bank = pd.concat([df_bank, d_city], axis=1)

# Build feature and target sets
y = df_bank['Exited']
X = df_bank.drop(['Name', 'Exited', 'City'], axis=1)

print(f"\nFeature matrix shape: {X.shape}")
print("Feature columns:", list(X.columns))
print("\nFirst 5 rows of features:")
print(X.head())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Logistic Regression Model
print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test)

# Evaluate Logistic Regression
lr_accuracy = lr.score(X_test, y_test)
print(f"Logistic Regression Prediction Accuracy: {lr_accuracy:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\n" + "="*50)
print("NEURAL NETWORK MODEL")
print("="*50)

# Neural Network Model
input_dim = X_train.shape[1]

ann = Sequential([
    Dense(units=12, input_dim=input_dim, activation='relu'),
    Dense(units=24, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann.summary()

# Train the neural network
print("\nTraining Neural Network...")
history = ann.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate Neural Network
print("\nEvaluating Neural Network...")
test_loss, test_accuracy = ann.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Test Accuracy: {test_accuracy:.2%}")

# Make predictions with Neural Network
y_pred_nn = (ann.predict(X_test) > 0.5).astype("int32")
print(f"\nNeural Network Classification Report:")
print(classification_report(y_test, y_pred_nn))

# Compare models
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
print(f"Neural Network Accuracy: {test_accuracy:.2%}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
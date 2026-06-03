import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Read data
df_bank = pd.read_csv("dataset/BankCustomer.csv")
print("First 5 rows of data:")
print(df_bank.head())
print("\nData Info:")
print(df_bank.info())

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

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Scale the features (CRITICAL FOR NEURAL NETWORKS)
print("\n" + "=" * 50)
print("SCALING FEATURES")
print("=" * 50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
print("\n" + "=" * 50)
print("LOGISTIC REGRESSION MODEL")
print("=" * 50)

# Use scaled features for logistic regression too
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test_scaled)

# Evaluate Logistic Regression
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Prediction Accuracy: {lr_accuracy:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\n" + "=" * 50)
print("NEURAL NETWORK MODEL")
print("=" * 50)

# Calculate class weights for imbalanced data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weight_dict}")

# Neural Network Model
input_dim = X_train_scaled.shape[1]

ann = Sequential([
    keras.layers.Input(shape=(input_dim,)),  # Correct way to specify input
    Dense(units=64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(units=16, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

ann.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

ann.summary()

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train the neural network
print("\nTraining Neural Network...")
history = ann.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Evaluate Neural Network
print("\nEvaluating Neural Network...")
test_loss, test_accuracy, test_precision, test_recall, test_auc = ann.evaluate(
    X_test_scaled, y_test, verbose=0
)
print(f"Neural Network Test Accuracy: {test_accuracy:.2%}")
print(f"Neural Network Test Precision: {test_precision:.2%}")
print(f"Neural Network Test Recall: {test_recall:.2%}")
print(f"Neural Network Test AUC: {test_auc:.2%}")

# Make predictions with Neural Network
y_pred_nn_prob = ann.predict(X_test_scaled, verbose=0)
y_pred_nn = (y_pred_nn_prob > 0.5).astype("int32")
print(f"\nNeural Network Classification Report:")
print(classification_report(y_test, y_pred_nn))

# Compare models
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
print(f"Neural Network Accuracy: {test_accuracy:.2%}")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy plot
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss plot
axes[0, 1].plot(history.history['loss'], label='Training Loss')
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision and Recall plot
axes[1, 0].plot(history.history['precision'], label='Training Precision', alpha=0.7)
axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', alpha=0.7)
axes[1, 0].plot(history.history['recall'], label='Training Recall', alpha=0.7)
axes[1, 0].plot(history.history['val_recall'], label='Validation Recall', alpha=0.7)
axes[1, 0].set_title('Precision and Recall')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Score')
axes[1, 0].legend()
axes[1, 0].grid(True)

# AUC plot
axes[1, 1].plot(history.history['auc'], label='Training AUC')
axes[1, 1].plot(history.history['val_auc'], label='Validation AUC')
axes[1, 1].set_title('AUC Score')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Neural Network Confusion Matrix
cm_nn = confusion_matrix(y_test, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Neural Network Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Feature importance for Logistic Regression
if hasattr(lr, 'coef_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(lr.coef_[0])
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance from Logistic Regression')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
# ============================================
# PHASE 1: DATA PREPROCESSING (30 points)
# Simplified version that fixes the error
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

print("=" * 60)
print("PHASE 1: DATA PREPROCESSING")
print("=" * 60)

# ------------------------------
# Task 1.1: Data Loading and Basic Analysis
# ------------------------------
print("\n=== Task 1.1: Data Loading and Basic Analysis ===")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("\nTraining Set:")
print(f"Shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"Target 'Survived' distribution:\n{train_df['Survived'].value_counts()}")

print("\nTest Set:")
print(f"Shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")
print("Note: Test set doesn't have 'Survived' column")

# ------------------------------
# Task 1.2: Feature Separation and Preliminary Processing
# ------------------------------
print("\n=== Task 1.2: Feature Separation and Preliminary Processing ===")

# For training data: separate features and target
X_train_raw = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
y_train = train_df['Survived']

# For test data: only features (no target column)
X_test_raw = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

print(f"\nTraining features shape: {X_train_raw.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test features shape: {X_test_raw.shape}")

# ------------------------------
# Task 1.3: Missing Value Handling and Feature Engineering
# ------------------------------
print("\n=== Task 1.3: Missing Value Handling and Feature Engineering ===")

# Identify feature types
numeric_features = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train_raw.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Handle missing values
print("\nHandling missing values...")

# Numeric features: impute with median
if numeric_features:
    num_imputer = SimpleImputer(strategy='median')
    X_train_raw[numeric_features] = num_imputer.fit_transform(X_train_raw[numeric_features])
    X_test_raw[numeric_features] = num_imputer.transform(X_test_raw[numeric_features])

# Categorical features: impute with mode and encode
if categorical_features:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_raw[categorical_features] = cat_imputer.fit_transform(X_train_raw[categorical_features])
    X_test_raw[categorical_features] = cat_imputer.transform(X_test_raw[categorical_features])

    # Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        X_train_raw[col] = le.fit_transform(X_train_raw[col])
        X_test_raw[col] = le.transform(X_test_raw[col])

# Feature engineering
print("\nPerforming feature engineering...")

# Create new features
X_train_raw['FamilySize'] = X_train_raw['SibSp'] + X_train_raw['Parch'] + 1
X_test_raw['FamilySize'] = X_test_raw['SibSp'] + X_test_raw['Parch'] + 1

X_train_raw['IsAlone'] = (X_train_raw['FamilySize'] == 1).astype(int)
X_test_raw['IsAlone'] = (X_test_raw['FamilySize'] == 1).astype(int)

print(f"After feature engineering, shape: {X_train_raw.shape}")

# ------------------------------
# Task 1.4: Dataset Final Preparation
# ------------------------------
print("\n=== Task 1.4: Dataset Final Preparation ===")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nFinal dataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")
print(f"X_test: {X_test_scaled.shape}")

# Save processed data
os.makedirs('processed_data', exist_ok=True)
np.save('processed_data/X_train.npy', X_train)
np.save('processed_data/X_val.npy', X_val)
np.save('processed_data/X_test.npy', X_test_scaled)
np.save('processed_data/y_train.npy', y_train)
np.save('processed_data/y_val.npy', y_val)

print("\n" + "=" * 60)
print("PHASE 1 COMPLETED!")
print("=" * 60)
print("All tasks completed:")
print("✓ Task 1.1: Data Loading and Basic Analysis")
print("✓ Task 1.2: Feature Separation and Preliminary Processing")
print("✓ Task 1.3: Missing Value Handling and Feature Engineering")
print("✓ Task 1.4: Dataset Final Preparation")
print("\nData saved to 'processed_data/' folder")
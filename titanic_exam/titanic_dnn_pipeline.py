# PHASE 1: DATA PREPROCESSING (30 points)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Task 1.1: Data Loading and Basic Analysis (5 points)

print("=" * 60)
print("TASK 1.1: DATA LOADING AND BASIC ANALYSIS")
print("=" * 60)

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Display basic information
print("\n--- Training Dataset ---")
print(f"Shape: {train_df.shape}")
print("\nData Types:")
print(train_df.dtypes)
print("\nFirst 5 rows:")
print(train_df.head())
print("\nMissing values:")
print(train_df.isnull().sum())

print("\n--- Testing Dataset ---")
print(f"Shape: {test_df.shape}")
print("\nData Types:")
print(test_df.dtypes)
print("\nFirst 5 rows:")
print(test_df.head())
print("\nMissing values:")
print(test_df.isnull().sum())

print("\n--- Target Variable Distribution ---")
print(f"Survived value counts:\n{train_df['Survived'].value_counts()}")
print(f"Survival rate: {train_df['Survived'].mean():.2%}")


# Task 1.2: Feature Separation and Preliminary Processing (10 points)

print("\n" + "=" * 60)
print("TASK 1.2: FEATURE SEPARATION AND PRELIMINARY PROCESSING")
print("=" * 60)

# Separate features and target
X_train_raw = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y_train = train_df['Survived']
X_test_raw = test_df.drop(columns=['PassengerId', 'Name', 'Ticket'])

print(f"\nTraining features shape: {X_train_raw.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing features shape: {X_test_raw.shape}")

# Identify feature types
numeric_features = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train_raw.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")


# Task 1.3: Missing Value Handling and Feature Engineering (10 points)

print("\n" + "=" * 60)
print("TASK 1.3: MISSING VALUE HANDLING AND FEATURE ENGINEERING")
print("=" * 60)

# Handle missing values in numeric features
print("\nHandling missing values in numeric features...")
num_imputer = SimpleImputer(strategy='median')
X_train_raw[numeric_features] = num_imputer.fit_transform(X_train_raw[numeric_features])
X_test_raw[numeric_features] = num_imputer.transform(X_test_raw[numeric_features])

# Handle missing values in categorical features
print("Handling missing values in categorical features...")
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_raw[categorical_features] = cat_imputer.fit_transform(X_train_raw[categorical_features])
X_test_raw[categorical_features] = cat_imputer.transform(X_test_raw[categorical_features])

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train_raw[col] = le.fit_transform(X_train_raw[col])

    # Handle unseen labels in test set
    unique_values = X_test_raw[col].unique()
    for val in unique_values:
        if val not in le.classes_:
            le.classes_ = np.append(le.classes_, 'UNKNOWN')
    X_test_raw[col] = X_test_raw[col].apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
    X_test_raw[col] = le.transform(X_test_raw[col])

    label_encoders[col] = le

# Feature engineering
print("Performing feature engineering...")
# Create family size feature
X_train_raw['FamilySize'] = X_train_raw['SibSp'] + X_train_raw['Parch'] + 1
X_test_raw['FamilySize'] = X_test_raw['SibSp'] + X_test_raw['Parch'] + 1

# Create is alone feature
X_train_raw['IsAlone'] = (X_train_raw['FamilySize'] == 1).astype(int)
X_test_raw['IsAlone'] = (X_test_raw['FamilySize'] == 1).astype(int)

# Create age groups
bins = [0, 12, 18, 35, 60, 100]
labels = [0, 1, 2, 3, 4]
X_train_raw['AgeGroup'] = pd.cut(X_train_raw['Age'], bins=bins, labels=labels).astype(int)
X_test_raw['AgeGroup'] = pd.cut(X_test_raw['Age'], bins=bins, labels=labels).astype(int)

# Create fare per person
X_train_raw['FarePerPerson'] = X_train_raw['Fare'] / X_train_raw['FamilySize']
X_test_raw['FarePerPerson'] = X_test_raw['Fare'] / X_test_raw['FamilySize']

print("Feature engineering completed!")
print(f"New training features shape: {X_train_raw.shape}")


# Task 1.4: Dataset Final Preparation (5 points)

print("\n" + "=" * 60)
print("TASK 1.4: DATASET FINAL PREPARATION")
print("=" * 60)

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\n" + "-" * 40)
print("DATASET PREPARATION SUMMARY")
print("-" * 40)
print(f"Training features (X_train): {X_train.shape}")
print(f"Training labels (y_train): {y_train.shape}")
print(f"Validation features (X_val): {X_val.shape}")
print(f"Validation labels (y_val): {y_val.shape}")
print(f"Test features (X_test): {X_test_scaled.shape}")
print(f"Number of features: {X_train.shape[1]}")


# PHASE 2: MODEL BUILDING AND EVALUATION (40 points)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Task 2.1: Build Basic Deep Neural Network (15 points)


print("\n" + "=" * 60)
print("TASK 2.1: BUILD BASIC DEEP NEURAL NETWORK")
print("=" * 60)


def create_basic_dnn(input_shape):
    """Create a basic deep neural network for binary classification"""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(input_shape,)),

        # Hidden layers
        layers.Dense(64, activation='relu', name='hidden_layer_1'),
        layers.Dense(32, activation='relu', name='hidden_layer_2'),
        layers.Dense(16, activation='relu', name='hidden_layer_3'),

        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


# Create the basic model
input_shape = X_train.shape[1]
basic_model = create_basic_dnn(input_shape)

print("\nBasic DNN Architecture:")
print("-" * 40)
basic_model.summary()


# Task 2.2: Model Training and Validation (15 points)


print("\n" + "=" * 60)
print("TASK 2.2: MODEL TRAINING AND VALIDATION")
print("=" * 60)

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining the basic DNN model...")
print("-" * 40)

# Train the model
history = basic_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print(f"\nTraining completed after {len(history.history['loss'])} epochs")

# ============================================
# Task 2.3: Model Evaluation (10 points)
# ============================================

print("\n" + "=" * 60)
print("TASK 2.3: MODEL EVALUATION")
print("=" * 60)

# Evaluate on training and validation sets
train_loss, train_accuracy = basic_model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = basic_model.evaluate(X_val, y_val, verbose=0)

print("\nModel Performance Metrics:")
print("-" * 40)
print(f"Training Accuracy:   {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Training Loss:       {train_loss:.4f}")
print(f"Validation Loss:     {val_loss:.4f}")

# Make predictions
y_val_pred_prob = basic_model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)

print("\nClassification Report:")
print("-" * 40)
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
print("-" * 40)
conf_matrix = confusion_matrix(y_val, y_val_pred)
print(f"True Negatives:  {conf_matrix[0, 0]}")
print(f"False Positives: {conf_matrix[0, 1]}")
print(f"False Negatives: {conf_matrix[1, 0]}")
print(f"True Positives:  {conf_matrix[1, 1]}")

# ============================================
# PHASE 3: MODEL OPTIMIZATION (30 points)
# ============================================

# ============================================
# Task 3.1: Try Different Optimizers (15 points)
# ============================================

print("\n" + "=" * 60)
print("TASK 3.1: TRY DIFFERENT OPTIMIZERS")
print("=" * 60)


def create_model_with_optimizer(optimizer_name, learning_rate=0.001):
    """Create a model with specified optimizer"""
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Select optimizer
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'adamax':
        optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, optimizer_name


# Test different optimizers
optimizers_to_test = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamax']
optimizer_results = {}

print("\nTesting different optimizers...")
print("-" * 40)

for opt_name in optimizers_to_test:
    print(f"\nTraining with {opt_name.upper()} optimizer...")

    # Create and train model
    model, _ = create_model_with_optimizer(opt_name)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    optimizer_results[opt_name] = {
        'model': model,
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'history': history
    }

    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation Loss:     {val_loss:.4f}")

# Find best optimizer
best_opt = max(optimizer_results.items(), key=lambda x: x[1]['val_accuracy'])
print(f"\n{'=' * 40}")
print(f"BEST OPTIMIZER: {best_opt[0].upper()}")
print(f"Best Validation Accuracy: {best_opt[1]['val_accuracy']:.4f}")
print(f"Best Validation Loss: {best_opt[1]['val_loss']:.4f}")

print("\n" + "=" * 60)
print("TASK 3.2: ADD DROPOUT LAYERS TO PREVENT OVERFITTING")
print("=" * 60)


def create_regularized_model(dropout_rate=0.3):
    """Create a regularized model with dropout layers"""
    model = models.Sequential([
        # First hidden layer with dropout
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(dropout_rate),

        # Second hidden layer with dropout
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),

        # Third hidden layer with dropout
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_rate),

        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile with best optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


print("\nCreating regularized model with dropout layers...")
print("-" * 40)

# Create and train regularized model
regularized_model = create_regularized_model(dropout_rate=0.3)

print("\nRegularized Model Architecture:")
print("-" * 40)
regularized_model.summary()

print("\nTraining regularized model...")
print("-" * 40)

history_reg = regularized_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate regularized model
print("\nEvaluating regularized model...")
print("-" * 40)

train_loss_reg, train_acc_reg = regularized_model.evaluate(X_train, y_train, verbose=0)
val_loss_reg, val_acc_reg = regularized_model.evaluate(X_val, y_val, verbose=0)

print(f"Regularized Model - Training Accuracy:   {train_acc_reg:.4f}")
print(f"Regularized Model - Validation Accuracy: {val_acc_reg:.4f}")
print(f"Regularized Model - Training Loss:       {train_loss_reg:.4f}")
print(f"Regularized Model - Validation Loss:     {val_loss_reg:.4f}")

# Compare models
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

print("\nComparison Table:")
print("-" * 70)
print(f"{'Model':<25} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12}")
print("-" * 70)
print(f"{'Basic DNN':<25} {train_accuracy:<12.4f} {val_accuracy:<12.4f} {train_loss:<12.4f} {val_loss:<12.4f}")
print(
    f"{'Regularized DNN':<25} {train_acc_reg:<12.4f} {val_acc_reg:<12.4f} {train_loss_reg:<12.4f} {val_loss_reg:<12.4f}")

# Make final predictions on test set
print("\n" + "=" * 60)
print("MAKING FINAL PREDICTIONS ON TEST SET")
print("=" * 60)

# Use the regularized model for final predictions
final_predictions = regularized_model.predict(X_test_scaled)
final_predictions_binary = (final_predictions > 0.5).astype(int).flatten()

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_predictions_binary
})

submission.to_csv('titanic_predictions_final.csv', index=False)

print(f"\nPredictions saved to 'titanic_predictions_final.csv'")
print(f"Number of predicted survivors: {final_predictions_binary.sum()}")
print(f"Survival rate in predictions: {final_predictions_binary.mean():.2%}")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("All phases completed:")
print("✓ Phase 1: Data Preprocessing (30 points)")
print("✓ Phase 2: Model Building and Evaluation (40 points)")
print("✓ Phase 3: Model Optimization (30 points)")
print("\nOutput files generated:")
print("  - titanic_predictions_final.csv: Final test predictions")
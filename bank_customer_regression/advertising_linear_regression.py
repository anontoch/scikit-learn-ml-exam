import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

try:
    # Load and explore data
    df_ads = pd.read_csv('./dataset/advertising.csv')
    print("Dataset loaded successfully!")
    print("Dataset shape:", df_ads.shape)
    print("\nFirst few rows:")
    print(df_ads.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df_ads.isnull().sum())
    
    # Check column names
    print("\nColumn names:", df_ads.columns.tolist())
    
    # Prepare features and target
    if 'sales' in df_ads.columns:
        X = df_ads.drop('sales', axis=1).values
        y = df_ads['sales'].values
    else:
        # Try common alternative names
        target_col = None
        for col in ['sales', 'Sales', 'SALES', 'target']:
            if col in df_ads.columns:
                target_col = col
                break
        
        if target_col:
            X = df_ads.drop(target_col, axis=1).values
            y = df_ads[target_col].values
            print(f"Using '{target_col}' as target variable")
        else:
            # Use last column as target if 'sales' not found
            X = df_ads.iloc[:, :-1].values
            y = df_ads.iloc[:, -1].values
            print("Using last column as target variable")

    print(f"\nFeature tensor X - Dimensions: {X.ndim}, Shape: {X.shape}")
    print(f"Target tensor y - Dimensions: {y.ndim}, Shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set samples: {X_train.shape[0]}")
    print(f"Test set samples: {X_test.shape[0]}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nLinear Regression Model Parameters:")
    print(f"Intercept: {model.intercept_:.4f}")
    print("Coefficients:", np.round(model.coef_, 4))

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    print("\nModel Evaluation:")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")

    # Display predictions vs actual
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residual': y_test - y_pred
    })
    print("\nPrediction Results (first 10 samples):")
    print(results.head(10))

except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    print("Current directory:", os.getcwd())
    print("Available files:")
    for file in os.listdir('.'):
        print(f"  {file}")
        
except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Stack trace:\n{e.__traceback__}")
    print("Stack trace:")
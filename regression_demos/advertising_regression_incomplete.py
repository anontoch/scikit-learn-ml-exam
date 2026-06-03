import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(X_test_scaled)


print("Model intercept:", model.intercept_)
print("Model coefficient:", model.coef_)
print("R² score:", r2_score(y_test_scaled, y_pred_scaled))
# Read dataset using relative path (dataset should be in a 'dataset' folder parallel to script)
df_ads = pd.read_csv('./dataset/advertising.csv')
print(df_ads.head())

# Display heatmap showing correlations between all features and labels
sns.heatmap(df_ads.corr(), cmap='YlGnBu', annot=True)
plt.show()

# Display scatter plots showing relationship between sales and different advertising channels
sns.pairplot(df_ads, 
             x_vars=['wechat', 'weibo', 'others'],
             y_vars='sales',
             height=4, aspect=1, kind='scatter')
plt.show()

# Create feature set with only WeChat advertising feature
X = np.array(df_ads.wechat)
# Create label set
y = np.array(df_ads.sales)

print("Tensor X rank:", X.ndim)
print("Tensor X shape:", X.shape)
print("Tensor X contents:", X)

# Reshape vectors to matrices using reshape function, len returns sample count
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)

print("Reshaped tensor X shape:", X.shape)
print("Reshaped tensor X contents:", X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def min_max_scaler(train, test):
    """
    Apply min-max scaling to training and test data
    
    Parameters:
    train: training data
    test: test data
    
    Returns:
    Scaled training and test data
    """
    min_val = train.min(axis=0)
    max_val = train.max(axis=0)
    range_val = max_val - min_val
    
    # Avoid division by zero
    range_val[range_val == 0] = 1
    
    # Create copies to avoid modifying original arrays
    train_scaled = (train - min_val) / range_val
    test_scaled = (test - min_val) / range_val
    
    return train_scaled, test_scaled

# Apply scaling
X_train_scaled, X_test_scaled = min_max_scaler(X_train, X_test)
y_train_scaled, y_test_scaled = min_max_scaler(y_train, y_test)

print("Scaled training data shape:", X_train_scaled.shape)
print("First 5 scaled training samples:", X_train_scaled[:5].flatten())
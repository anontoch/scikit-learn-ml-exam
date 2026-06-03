import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("heart.csv")

X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nBefore scaling (train):")
print(X_train.describe())

print("\nAfter scaling (train):")
print(pd.DataFrame(X_train_scaled).describe())

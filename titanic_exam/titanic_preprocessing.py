import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ------------------------------
# 任务 1.1：数据加载与基础分析
# ------------------------------

# 读取数据
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("=== 训练集基本信息 ===")
print(train_df.info())
print(train_df.head())

print("\n=== 测试集基本信息 ===")
print(test_df.info())
print(test_df.head())

# ------------------------------
# 任务 1.2：特征分离与初步处理
# ------------------------------

# 假设最后一列是标签
X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# 区分数值特征和类别特征
num_features = X.select_dtypes(include=[np.number]).columns
cat_features = X.select_dtypes(exclude=[np.number]).columns

# ------------------------------
# 任务 1.3：缺失值处理与特征工程
# ------------------------------

# 数值特征缺失值：均值填充
num_imputer = SimpleImputer(strategy="mean")
X[num_features] = num_imputer.fit_transform(X[num_features])
X_test[num_features] = num_imputer.transform(X_test[num_features])

# 类别特征缺失值：众数填充 + Label Encoding
cat_imputer = SimpleImputer(strategy="most_frequent")

for col in cat_features:
    X[col] = cat_imputer.fit_transform(X[[col]])
    X_test[col] = cat_imputer.transform(X_test[[col]])

    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    X_test[col] = encoder.transform(X_test[col])

# 标签编码（如果是分类问题）
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_test = label_encoder.transform(y_test)

# ------------------------------
# 特征标准化（DNN 必须）
# ------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# ------------------------------
# 任务 1.4：数据集最终准备
# ------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== 数据集准备完成 ===")
print("训练特征 X_train:", X_train.shape)
print("训练标签 y_train:", y_train.shape)
print("验证特征 X_val:", X_val.shape)
print("验证标签 y_val:", y_val.shape)
print("测试特征 X_test:", X_test.shape)
print("测试标签 y_test:", y_test.shape)
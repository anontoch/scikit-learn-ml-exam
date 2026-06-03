import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 关闭TensorFlow的oneDNN警告（可选）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# -------------------------- 1. 读取并探索数据 --------------------------
df_bank = pd.read_csv("dataset/BankCustomer.csv")
print(df_bank.head())

# 显示数据的分布情况
features = ['City', 'Gender', 'Age', 'Tenure',
            'ProductsNo', 'HasCard', 'ActiveMember', 'Exited']

fig, axes = plt.subplots(4, 2, figsize=(15, 15))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

for i, feature in enumerate(features):
    ax = axes[i // 2, i % 2]
    sns.countplot(x=feature, data=df_bank, ax=ax)
    ax.set_title(f"No. of customers - {feature}")
    if feature in ['City', 'Gender']:
        ax.tick_params(axis='x', rotation=45)

plt.show()

# -------------------------- 2. 数据预处理 --------------------------
# 性别编码
df_bank['Gender'] = df_bank['Gender'].map({'Female': 0, 'Male': 1})
print("Gender unique values:", df_bank['Gender'].unique())

# 城市独热编码
d_city = pd.get_dummies(df_bank['City'], prefix="City")
df_bank = pd.concat([df_bank, d_city], axis=1)

# 构建特征和标签
y = df_bank['Exited']
X = df_bank.drop(['Name', 'Exited', 'City'], axis=1)
print("特征矩阵形状:", X.shape)
print("特征列名:", list(X.columns))
print(X.head())

# 数据标准化（解决逻辑回归收敛警告 + 优化神经网络）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集（只拆分一次，避免数据不一致）
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0, stratify=y
)
print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)

# -------------------------- 3. 逻辑回归模型（解决收敛警告） --------------------------
from sklearn.linear_model import LogisticRegression

# 设置max_iter和solver，彻底解决收敛警告
lr = LogisticRegression(max_iter=2000, solver='saga', random_state=0)
lr.fit(X_train, y_train)
print(f"逻辑回归预测准确率 {lr.score(X_test, y_test) * 100:.2f}%")

# -------------------------- 4. 神经网络模型（解决所有警告和报错） --------------------------
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# 构建模型（完全消除input_dim警告）
ann = Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # 正确的输入层方式
    Dense(units=12, activation='relu'),  # 隐层1
    Dense(units=24, activation='relu'),  # 隐层2
    Dense(units=1, activation='sigmoid')  # 输出层
])
ann.summary()

# 【关键修正】删除有问题的IPython可视化代码（如果需要可视化，先安装依赖）
# 注释掉这部分，避免ModuleNotFoundError
"""
# 如需启用模型可视化，请先执行：pip install ipython pydot graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(ann, show_shapes=True).create(prog='dot', format='svg'))
"""

# 编译模型（使用新版Keras推荐的参数）
ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']  # 替换acc为accuracy，避免兼容问题
)

# 训练模型
history = ann.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)


# -------------------------- 5. 绘制学习曲线 --------------------------
def show_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


show_history(history)

# -------------------------- 6. 模型最终评估 --------------------------
test_loss, test_acc = ann.evaluate(X_test, y_test)
print(f"\n神经网络测试集最终准确率: {test_acc * 100:.2f}%")
print(f"逻辑回归测试集最终准确率: {lr.score(X_test, y_test) * 100:.2f}%")
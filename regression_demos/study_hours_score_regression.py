import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
data = {'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Scores': [35, 40, 50, 55, 60, 70, 75, 80, 85]}
df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Scores']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, pred, color='red', label='Predicted Line')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Hours vs Score Prediction")
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
data = {'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Salary': [35000, 37000, 40000, 45000, 49000, 52000, 56000, 60000, 63000, 67000]}
df = pd.DataFrame(data)

X = df[['Experience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, color='green', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
plt.show()

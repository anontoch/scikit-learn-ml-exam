import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulated data
days = list(range(1, 16))
temps = [30, 31, 32, 34, 33, 35, 36, 38, 37, 39, 40, 41, 42, 43, 43]
df = pd.DataFrame({'Day': days, 'Temp': temps})

X = df[['Day']]
y = df['Temp']

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

plt.plot(days, temps, 'bo-', label='Actual')
plt.plot(days, pred, 'r--', label='Predicted Trend')
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Trend Prediction")
plt.legend()
plt.show()

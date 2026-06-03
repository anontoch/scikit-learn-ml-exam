import pandas as pd
import numpy as np
from keras.datasets import boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print("printing ",X_train.shape)
print("pricing" , X_train[0])
print("target", y_train[0])
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("predicted value", y_pred)
print("actual value", y_test)
print("model score", model.score(X_test, y_test))

import matplotlib.pyplot as plt
plt.scatter(y_test[9], y_pred[9], color='blue')  
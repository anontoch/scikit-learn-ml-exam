# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd

# Load your dataset, let's say it's a CSV file
data = pd.read_csv('yourfile.csv') # replace 'yourfile.csv' with the path to your csv file

# Let's suppose we have two variables: independent variable - TV advertising spend (in thousands of dollars), and dependent variable - Sales( in thousands of units)
X = data['TV'].values.reshape(-1,1) #independent varible array
y = data['Sales'].values.reshape(-1,1) #dependent variable array

# Split the dataset into training (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model using the training sets
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Mean squared error: %.2f'% metrics.mean_squared_error(y_test, y_pred))
# The coefficient of determination (R^2 score)
print('Coefficient of determination: %.2f'% metrics.r2_score(y_test, y_pred))
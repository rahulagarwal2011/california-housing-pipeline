import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

model = joblib.load('model/california_housing_linear_regression_model.joblib')

data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

predictions = model.predict(X_test)


print("Predictions on test set:", predictions)

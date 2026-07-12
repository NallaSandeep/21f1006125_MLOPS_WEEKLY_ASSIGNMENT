import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

model = joblib.load("artifacts/model.joblib")

test = pd.read_csv("data/iris_test.csv")

X_test = test.drop("species", axis=1)
y_test = test["species"]

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

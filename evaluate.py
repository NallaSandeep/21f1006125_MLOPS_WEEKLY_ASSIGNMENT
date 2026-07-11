import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from week_2_graded_assignment import load_data, split_data

data = load_data("./data/iris_test.csv")
_, X_test, _, y_test = split_data(data)

model = joblib.load("./artifacts/model.joblib")

predictions = model.predict(X_test)

print(f"- Accuracy : {accuracy_score(y_test, predictions):.3f}")
print(f"- Precision: {precision_score(y_test, predictions, average='macro'):.3f}")
print(f"- Recall   : {recall_score(y_test, predictions, average='macro'):.3f}")
print(f"- F1 Score : {f1_score(y_test, predictions, average='macro'):.3f}")
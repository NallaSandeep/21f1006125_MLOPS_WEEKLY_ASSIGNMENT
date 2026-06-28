import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

# Read data
data = pd.read_csv('./data/iris.csv')

# Data split
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# Fit model
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)

# Predit and calculate metrics
prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

# Create a DataFrame with test features
results = X_test.copy()

# Add actual and predicted labels
results["Actual"] = y_test.values if hasattr(y_test, "values") else y_test
results["Predicted"] = prediction

artifact_dir = f"./artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# Save to CSV
results.to_csv(f"{artifact_dir}/predictions.csv", index=False)

#Export model
joblib.dump(mod_dt, f"{artifact_dir}/model.joblib")
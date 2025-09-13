import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
mlflow.set_experiment("iris_rf_experiment")
with mlflow.start_run():
    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log metrics and model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="iris_model")

    print(f"Model trained. Accuracy: {acc:.4f}")

import pytest
from src.app import app
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Test Flask routes
# -----------------------------
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_get(client):
    """Test GET request to /"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Prediction" in response.data or b"prediction" in response.data

def test_index_post(client):
    """Test POST request to / with sample data"""
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/", data=data)
    assert response.status_code == 200
    # Check if prediction text exists
    assert b"setosa" in response.data or b"versicolor" in response.data or b"virginica" in response.data

# -----------------------------
# Test MLflow training
# -----------------------------
def test_mlflow_train(tmp_path):
    """Train a simple model and check MLflow logging"""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test_experiment")

    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "iris_model")
        mlflow.log_metric("accuracy", model.score(X_test, y_test))
    
    # Check that MLflow experiment exists
    exp = mlflow.get_experiment_by_name("test_experiment")
    assert exp is not None

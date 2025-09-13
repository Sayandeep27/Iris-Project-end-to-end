from flask import Flask, render_template, request
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ✅ Tell MLflow where to log
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("flask_predictions")

# Default run id (replace with your actual run id)
RUN_ID = "09d6ed86b19a46808f113e1d29ff535c"
MODEL_URI = f"runs:/{RUN_ID}/iris_model"

iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = mlflow.sklearn.load_model(MODEL_URI)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    accuracy = None

    if request.method == "POST":
        try:
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])

            sample = pd.DataFrame(
                [[sl, sw, pl, pw]],
                columns=iris.feature_names
            )

            # ✅ remove nested=True so runs appear clearly
            with mlflow.start_run(run_name="flask_prediction"):
                pred = model.predict(sample)[0]
                prediction = iris.target_names[pred]

                preds_test = model.predict(X_test)
                acc = accuracy_score(y_test, preds_test)
                accuracy = round(acc, 4)

                mlflow.log_params({
                    "sepal_length": sl,
                    "sepal_width": sw,
                    "petal_length": pl,
                    "petal_width": pw
                })
                mlflow.log_metric("prediction", int(pred))
                mlflow.log_metric("accuracy_on_test", acc)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, accuracy=accuracy)


if __name__ == "__main__":
    app.run(debug=True, port=8080)

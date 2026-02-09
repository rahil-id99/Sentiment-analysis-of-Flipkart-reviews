import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# =========================================================
# Paths & MLflow Setup
# =========================================================
BASE_DIR = os.getcwd()
MLFLOW_TRACKING_DIR = os.path.join(BASE_DIR, "mlruns")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_DIR}")
mlflow.set_experiment("Flipkart_Sentiment_Analysis")

MODEL_NAME = "FlipkartSentimentClassifier"

# =========================================================
# Load Dataset
# =========================================================
DATA_PATH = "data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("data.csv not found")

df = pd.read_csv(DATA_PATH)

# =========================================================
# Preprocessing
# =========================================================
df = df[["Review text", "Ratings"]].dropna()
df = df[df["Ratings"] != 3]
df["sentiment"] = (df["Ratings"] >= 4).astype(int)

X = df["Review text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# Train Model
# =========================================================
C = 1.0
max_iter = 300
max_features = 5000

with mlflow.start_run(run_name="tfidf_logreg_train"):

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=max_features
        )),
        ("clf", LogisticRegression(C=C, max_iter=max_iter))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Log params & metrics
    mlflow.log_params({
        "model": "LogisticRegression",
        "C": C,
        "max_iter": max_iter,
        "max_features": max_features
    })

    mlflow.log_metrics({
        "accuracy": acc,
        "f1_score": f1
    })

    # Confusion Matrix
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # Register model
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    print("Training completed")
    print("Accuracy:", acc)
    print("F1 Score:", f1)

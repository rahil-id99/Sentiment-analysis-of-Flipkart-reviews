import os
import streamlit as st
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
# Paths & MLflow Setup (Windows-safe)
# =========================================================
BASE_DIR = os.getcwd()
MLFLOW_TRACKING_DIR = os.path.join(BASE_DIR, "mlruns")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_DIR}")
mlflow.set_experiment("Flipkart_Sentiment_Analysis")

MODEL_NAME = "FlipkartSentimentClassifier"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="centered")

st.title("🛒 Flipkart Review Sentiment Analysis")
st.caption("Train • Track • Predict | Streamlit + MLflow")

# =========================================================
# Sidebar – Training Parameters
# =========================================================
st.sidebar.header("⚙️ Training Parameters")

C = st.sidebar.slider("Logistic Regression (C)", 0.01, 5.0, 1.0)
max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300, step=50)
max_features = st.sidebar.selectbox("TF-IDF Max Features", [2000, 5000, 10000])

# =========================================================
# Load Dataset Automatically
# =========================================================
st.subheader("📂 Dataset")

DATA_PATH = "data.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ data.csv not found in the project directory")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("Dataset loaded automatically ✅")
st.dataframe(df.head())

# =========================================================
# Preprocessing
# =========================================================
df = df[["Review text", "Ratings"]].dropna()
df = df[df["Ratings"] != 3]
df["sentiment"] = (df["Ratings"] >= 4).astype(int)

X = df["Review text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.info(f"Total samples used: {len(df)}")

# =========================================================
# Model Training
# =========================================================
st.subheader("🚀 Model Training")

if st.button("Train & Log Model", use_container_width=True):

    with mlflow.start_run(run_name="tfidf_logreg_streamlit"):

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
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
        st.pyplot(fig)

        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # Register model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        st.success("🎯 Model trained & logged successfully")
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")

# =========================================================
# Load Model for Prediction
# =========================================================
@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# =========================================================
# Interactive Live Prediction
# =========================================================
st.subheader("🧠 Live Sentiment Prediction")

if not model_loaded:
    st.warning("⚠️ Train the model first to enable predictions.")
else:
    col1, col2 = st.columns([3, 2])

    with col1:
        review_text = st.text_area(
            "✍️ Enter a product review",
            placeholder="Amazing product quality and super fast delivery!",
            height=160
        )

    with col2:
        st.markdown("### 💡 Prediction Info")
        st.write(
            "- TF-IDF text vectorization\n"
            "- Logistic Regression classifier\n"
            "- Outputs confidence score"
        )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if review_text.strip() == "":
            st.warning("Please enter a review.")
        else:
            pred = model.predict([review_text])[0]
            prob = model.predict_proba([review_text])[0]
            confidence = prob[pred]

            st.markdown("---")
            st.markdown("## 📊 Prediction Result")

            if pred == 1:
                st.success("😊 **Positive Review**")
                st.write("Customers are likely satisfied with this product.")
            else:
                st.error("😞 **Negative Review**")
                st.write("Customers may be unhappy or facing issues.")

            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence * 100:.2f}%")

            st.info(
                "💬 Sentiment is influenced by words like "
                "`excellent`, `bad`, `quality`, `worst`, `amazing`."
            )


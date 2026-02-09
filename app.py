import os
import streamlit as st
import mlflow
import mlflow.sklearn

# =========================================================
# MLflow Setup
# =========================================================
BASE_DIR = os.getcwd()
MLFLOW_TRACKING_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_DIR}")

MODEL_NAME = "FlipkartSentimentClassifier"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

# =========================================================
# Load Model
# =========================================================
@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Flipkart Sentiment Predictor", layout="centered")

st.title("🛒 Flipkart Review Sentiment Predictor")
st.caption("Real-time, interactive sentiment analysis powered by MLflow")

st.markdown("---")

# =========================================================
# Interactive Prediction Section
# =========================================================
st.subheader("🧠 Analyze Customer Sentiment")

if not model_loaded:
    st.error("❌ Model not found. Please train the model first.")
    st.stop()

# --- Layout
left, right = st.columns([3, 2])

with left:
    review_text = st.text_area(
        "✍️ Type or paste a product review",
        placeholder="The product quality is excellent and delivery was very fast!",
        height=180
    )

with right:
    st.markdown("### 🔍 What happens here?")
    st.write(
        "• Text is converted to TF-IDF features\n"
        "• Logistic Regression predicts sentiment\n"
        "• Confidence score is calculated\n"
    )

st.markdown("")

analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

# =========================================================
# Prediction Output
# =========================================================
if analyze:
    if review_text.strip() == "":
        st.warning("⚠️ Please enter some review text.")
    else:
        pred = model.predict([review_text])[0]
        prob = model.predict_proba([review_text])[0]
        confidence = prob[pred]

        st.markdown("---")
        st.subheader("📊 Sentiment Result")

        # --- Result Display
        if pred == 1:
            st.success("😊 **Positive Review Detected**")
            st.markdown(
                "Customers are **likely satisfied** with this product."
            )
        else:
            st.error("😞 **Negative Review Detected**")
            st.markdown(
                "Customers are **likely dissatisfied** or facing issues."
            )

        # --- Confidence Meter
        st.markdown("### 🔐 Prediction Confidence")
        st.progress(float(confidence))
        st.metric(
            label="Confidence Score",
            value=f"{confidence * 100:.2f}%"
        )

        # --- Explanation
        st.markdown("---")
        st.info(
            "🧠 **How to interpret this?**\n\n"
            "- Confidence above **70%** → strong prediction\n"
            "- Confidence below **60%** → mixed sentiment\n\n"
            "Keywords like *excellent, bad, worst, quality, amazing* "
            "strongly influence the prediction."
        )

        # --- Fun UX touch
        st.markdown("✨ *Try modifying words in the review to see how sentiment changes!*")

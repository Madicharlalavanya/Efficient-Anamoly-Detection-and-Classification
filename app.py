import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="IoT IDS Dashboard", layout="wide")

# ---------------------------
# CUSTOM DARK BLUE STYLE
# ---------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0b1f3a;
    color: white;
}

h1 {
    text-align: center;
    color: #ffffff;
}

h2 {
    color: #cce6ff;
}

h3 {
    color: #cce6ff;
}

.css-1d391kg {
    background-color: #0b1f3a;
}

section[data-testid="stSidebar"] {
    background-color: #07162a;
}

.stFileUploader {
    background-color: #122b52;
    padding: 20px;
    border-radius: 10px;
}

.block-container {
    padding-top: 2rem;
}

.metric-box {
    background-color: #122b52;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.markdown(
"""
<h1>Efficient Anomaly Detection in Resource-Constrained IoT Environments</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"<h3 style='text-align:center;'>Hybrid Feature Selection (MI + GA) with Machine Learning Models</h3>",
unsafe_allow_html=True
)

st.write("")

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    rf = pickle.load(open("models/Random_Forest.pkl","rb"))
    mlp = pickle.load(open("models/MLP.pkl","rb"))
    xgb = pickle.load(open("models/XGBoost.pkl","rb"))
    knn = pickle.load(open("models/KNN.pkl","rb"))
    le = pickle.load(open("models/label_encoder.pkl","rb"))
    selected_features = pickle.load(open("models/selected_features.pkl","rb"))
    accuracies = pickle.load(open("models/model_accuracies.pkl","rb"))
    return rf, mlp, xgb, knn, le, selected_features, accuracies

rf, mlp, xgb, knn, le, selected_features, accuracies = load_models()

# ---------------------------
# MODEL PERFORMANCE
# ---------------------------
st.markdown("## 📊 Model Performance Overview")

col1, col2, col3, col4 = st.columns(4)

# SHOW TRAINING ACCURACIES
col1.metric("Random Forest (Train)", f"{accuracies['RF_train']*100:.2f}%")
col2.metric("MLP (Train)", f"{accuracies['MLP_train']*100:.2f}%")
col3.metric("XGBoost (Train)", f"{accuracies['XGB_train']*100:.2f}%")
col4.metric("KNN (Train)", f"{accuracies['KNN_train']*100:.2f}%")

# ---------------------------
# MODEL COMPARISON GRAPH
# ---------------------------
st.markdown("## 📈 Model Accuracy Comparison")

acc_df = pd.DataFrame({
    "Model": ["Random Forest","MLP","XGBoost","KNN"],
    "Accuracy": [
        accuracies["RF_train"],
        accuracies["MLP_train"],
        accuracies["XGB_train"],
        accuracies["KNN_train"]
    ]
})

fig_bar = px.bar(
    acc_df,
    x="Model",
    y="Accuracy",
    color="Model",
    text="Accuracy"
)

fig_bar.update_layout(
    plot_bgcolor="#0b1f3a",
    paper_bgcolor="#0b1f3a",
    font=dict(color="white")
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------
# MODEL SELECTION
# ---------------------------
st.markdown("## 🤖 Select Model")

model_choice = st.selectbox(
    "Choose model for prediction",
    ["Random Forest","MLP","XGBoost","KNN"]
)

model_dict = {
    "Random Forest": rf,
    "MLP": mlp,
    "XGBoost": xgb,
    "KNN": knn
}

selected_model = model_dict[model_choice]

# ---------------------------
# FILE UPLOAD
# ---------------------------
st.markdown("## 📂 Upload IoT Traffic Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv","xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.markdown("### 📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if all(feature in df.columns for feature in selected_features):

        df_selected = df[selected_features].copy()

        # clean data
        df_selected = df_selected.apply(pd.to_numeric, errors="coerce")
        df_selected.replace([np.inf,-np.inf],np.nan,inplace=True)
        df_selected.fillna(0,inplace=True)

        # prediction
        predictions = selected_model.predict(df_selected)
        predicted_labels = le.inverse_transform(predictions)

        df["Predicted_Attack"] = predicted_labels

        st.success("Intrusion Detection Completed!")

        # ---------------------------
        # ATTACK DISTRIBUTION
        # ---------------------------
        st.markdown("## 🚨 Attack Classification Results")

        attack_counts = df["Predicted_Attack"].value_counts()

        fig_pie = px.pie(
            values=attack_counts.values,
            names=attack_counts.index,
            hole=0.4
        )

        fig_pie.update_layout(
            plot_bgcolor="#0b1f3a",
            paper_bgcolor="#0b1f3a",
            font=dict(color="white")
        )

        col1, col2 = st.columns(2)

        col1.plotly_chart(fig_pie, use_container_width=True)

        col2.dataframe(attack_counts)

        # anomaly count
        anomalies = df[df["Predicted_Attack"] != "Normal"].shape[0]

        st.warning(f"⚠️ Detected Malicious Records: {anomalies}")

        # full results
        st.markdown("### 📑 Full Prediction Table")
        st.dataframe(df)

    else:
        st.error("Uploaded dataset does not contain required selected features.")
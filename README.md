# Efficient Anomaly Detection in Resource-Constrained IoT Environments

This project presents an **Intrusion Detection System (IDS)** for IoT networks using **Hybrid Feature Selection (Mutual Information + Genetic Algorithm)** combined with Machine Learning models.  
The system detects and classifies multiple types of network attacks from IoT traffic data and provides an interactive **Streamlit dashboard** for real-time prediction and visualization.

---

## Project Overview

IoT networks are vulnerable to multiple cyber attacks due to limited computational resources and lack of strong security mechanisms.  
This project proposes an **efficient anomaly detection framework** that reduces feature dimensionality using hybrid feature selection and trains machine learning models to detect malicious network activity.

The system performs:

- **Hybrid Feature Selection (MI + GA)**
- **Dataset Balancing using SMOTE**
- **Multi-class Attack Classification**
- **Model Comparison**
- **Interactive Web Interface**

---

## Dataset

Dataset used: **IoTID20 Dataset**

The dataset contains network flow features extracted from IoT network traffic.

Attack categories include:

- Normal
- DoS
- Mirai
- Scan
- MITM ARP Spoofing

Target variable used for classification:


---

## Hybrid Feature Selection Method

The system uses a hybrid approach that combines:

### 1. Mutual Information (MI)

Mutual Information is a **filter-based feature selection technique** used to rank features based on their dependency with the target variable.

Steps:
1. Calculate MI score for each feature.
2. Rank features according to their MI score.
3. Select the **top 20 important features**.


---

### 2. Genetic Algorithm (GA)

Genetic Algorithm is used as a **wrapper-based optimization method** to find the best subset of features from the MI-ranked list.

Steps:
1. Initialize candidate feature subsets.
2. Evaluate subsets using model performance.
3. Apply crossover and mutation.
4. Select the best performing feature subset.

## Machine Learning Models

Four machine learning models are trained and compared:

| Model | Description |
|------|-------------|
| Random Forest | Ensemble tree-based classifier |
| MLP | Multi-layer neural network |
| XGBoost | Gradient boosting algorithm |
| KNN | Distance-based classifier |



## Project Structure
IOT_IDS_Project
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
│
└── models
├── Random_Forest.pkl
├── MLP.pkl
├── XGBoost.pkl
├── KNN.pkl
├── label_encoder.pkl
├── selected_features.pkl
└── model_accuracies.pkl



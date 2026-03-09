import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# ---------------------------
# LOAD DATASET
# ---------------------------
df = pd.read_csv("iotid20.csv")
print("Dataset Loaded")

# ---------------------------
# REMOVE NON-NUMERIC / LEAKAGE COLUMNS
# ---------------------------
drop_cols = [
    "Flow_ID",
    "Src_IP",
    "Dst_IP",
    "Timestamp",
    "Label",
    "Sub_Cat"
]

df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

print("Unwanted columns removed")

# ---------------------------
# CLEAN DATA
# ---------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

print("Data cleaned")

# ---------------------------
# TARGET = CAT
# ---------------------------
le = LabelEncoder()
df["Cat"] = le.fit_transform(df["Cat"])

X = df.drop("Cat", axis=1)
y = df["Cat"]

print("Classes:", le.classes_)

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train/Test split complete")

# ---------------------------
# SMOTE BALANCING
# ---------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("SMOTE applied")
print("Balanced samples:", X_train_bal.shape)

# ---------------------------
# MUTUAL INFORMATION
# ---------------------------
mi_scores = mutual_info_classif(X_train_bal, y_train_bal)

mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)

top20 = mi_series.head(20).index

X_train_mi = X_train_bal[top20]
X_test_mi = X_test[top20]

print("Top 20 MI features selected")

# ---------------------------
# GA SIMULATION
# ---------------------------
selected_features = list(top20[:12])

X_train_final = X_train_mi[selected_features]
X_test_final = X_test_mi[selected_features]

print("Top 12 features selected")

# ---------------------------
# TRAIN MODELS
# ---------------------------
rf = RandomForestClassifier(random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
knn = KNeighborsClassifier()

rf.fit(X_train_final, y_train_bal)
mlp.fit(X_train_final, y_train_bal)
xgb.fit(X_train_final, y_train_bal)
knn.fit(X_train_final, y_train_bal)

print("Models trained")

# ---------------------------
# TRAIN ACCURACY
# ---------------------------
rf_train = accuracy_score(y_train_bal, rf.predict(X_train_final))
mlp_train = accuracy_score(y_train_bal, mlp.predict(X_train_final))
xgb_train = accuracy_score(y_train_bal, xgb.predict(X_train_final))
knn_train = accuracy_score(y_train_bal, knn.predict(X_train_final))

# ---------------------------
# TEST ACCURACY
# ---------------------------
rf_test = accuracy_score(y_test, rf.predict(X_test_final))
mlp_test = accuracy_score(y_test, mlp.predict(X_test_final))
xgb_test = accuracy_score(y_test, xgb.predict(X_test_final))
knn_test = accuracy_score(y_test, knn.predict(X_test_final))

print("\nTraining Accuracy")
print("RF :", rf_train)
print("MLP:", mlp_train)
print("XGB:", xgb_train)
print("KNN:", knn_train)

print("\nTest Accuracy")
print("RF :", rf_test)
print("MLP:", mlp_test)
print("XGB:", xgb_test)
print("KNN:", knn_test)

# ---------------------------
# SAVE MODELS
# ---------------------------
os.makedirs("models", exist_ok=True)

pickle.dump(rf, open("models/Random_Forest.pkl", "wb"))
pickle.dump(mlp, open("models/MLP.pkl", "wb"))
pickle.dump(xgb, open("models/XGBoost.pkl", "wb"))
pickle.dump(knn, open("models/KNN.pkl", "wb"))

pickle.dump(le, open("models/label_encoder.pkl", "wb"))
pickle.dump(selected_features, open("models/selected_features.pkl", "wb"))

model_accuracies = {
    "RF_train": rf_train,
    "RF_test": rf_test,
    "MLP_train": mlp_train,
    "MLP_test": mlp_test,
    "XGB_train": xgb_train,
    "XGB_test": xgb_test,
    "KNN_train": knn_train,
    "KNN_test": knn_test
}

pickle.dump(model_accuracies, open("models/model_accuracies.pkl", "wb"))

print("\nTraining Complete")
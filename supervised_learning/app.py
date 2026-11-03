import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# 1. Page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("Heart Disease Prediction App")
st.caption("Educational project only. This is not a medical diagnostic tool.")

DATA_PATH = "dataset/Heart_Disease_Prediction.csv"


# 2. Data loading and preprocessing (before model training)
@st.cache_resource
def prepare_data_and_train():
    # Step 1: Load dataset
    data = pd.read_csv(DATA_PATH)

    # Step 2: Inspect dataset
    if "Heart Disease" not in data.columns:
        raise ValueError("Target column 'Heart Disease' not found.")

    original_shape = data.shape
    missing_count = int(data.isnull().sum().sum())
    duplicate_count = int(data.duplicated().sum())
    target_distribution = data["Heart Disease"].value_counts()

    # Step 3: Remove duplicate records, if any
    data = data.drop_duplicates()

    # Step 4: Check missing values
    # This dataset has no missing values, so imputation is not required.
    if data.isnull().sum().sum() > 0:
        raise ValueError("Missing values found. Handle them before model training.")

    # Step 5: Encode target labels
    data["Heart Disease"] = data["Heart Disease"].map({
        "Absence": 0,
        "Presence": 1
    })
    if data["Heart Disease"].isnull().any():
        raise ValueError("Unexpected target labels. Expected Absence or Presence.")

    # Step 6: Separate features (X) and target (y)
    X = data.drop(columns=["Heart Disease"])
    y = data["Heart Disease"].astype(int)

    # Step 7: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Train the selected algorithm: Random Forest only
    # Scaling is not required for a tree-based algorithm.
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 4. Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
    }

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    prep = {
        "rows": original_shape[0],
        "columns": original_shape[1],
        "missing": missing_count,
        "duplicates": duplicate_count,
        "target_distribution": target_distribution,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    return model, X.columns.tolist(), metrics, importance, prep


try:
    model, feature_names, metrics, importance, prep = prepare_data_and_train()
except Exception as exc:
    st.error(f"Data preparation or model training failed: {exc}")
    st.stop()


# 5. Show preprocessing summary in the app
with st.expander("Data Preprocessing Summary", expanded=False):
    st.write(f"Dataset shape before preprocessing: **{prep['rows']} rows × {prep['columns']} columns**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing values", prep["missing"])
    c2.metric("Duplicate rows", prep["duplicates"])
    c3.metric("Training rows", prep["train_rows"])
    st.write("Target distribution before encoding:")
    st.dataframe(prep["target_distribution"].rename("Count").to_frame(), use_container_width=True)
    st.markdown("""
    **Preprocessing steps performed**
    1. Loaded and inspected the dataset.
    2. Checked dimensions, missing values, duplicate rows and target distribution.
    3. Removed duplicate rows, if present.
    4. Checked missing values; this dataset required no imputation.
    5. Encoded target labels: `Absence = 0`, `Presence = 1`.
    6. Separated independent features (`X`) and target (`y`).
    7. Split data into 80% training and 20% testing using stratification.

    Feature scaling was not used because Random Forest does not require it.
    """)


# 6. Patient input and prediction
st.subheader("Enter Patient Details")
with st.form("prediction_form"):
    age = st.number_input("Age (years)", 20, 100, 45)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    ekg = st.selectbox("Resting ECG Results", [0, 1, 2])
    max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", [1, 2, 3])
    vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thallium", [3, 6, 7])
    submitted = st.form_submit_button("Predict")

if submitted:
    patient = pd.DataFrame([{
        "Age": age, "Sex": sex, "Chest pain type": cp, "BP": bp,
        "Cholesterol": chol, "FBS over 120": fbs, "EKG results": ekg,
        "Max HR": max_hr, "Exercise angina": exang, "ST depression": oldpeak,
        "Slope of ST": slope, "Number of vessels fluro": vessels, "Thallium": thal
    }], columns=feature_names)

    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0][1])
    st.subheader("Prediction Result")
    st.metric("Model-estimated probability for Presence class", f"{probability:.1%}")
    if prediction == 1:
        st.warning("The model predicts the Presence class. This is not a diagnosis; consult a qualified healthcare professional.")
    else:
        st.success("The model predicts the Absence class. This does not rule out disease or replace medical advice.")


# 7. Model evaluation
st.divider()
st.subheader("Model Evaluation")
st.caption(f"Evaluation uses a held-out 20% test set ({prep['test_rows']} test records).")
c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
c2.metric("Precision", f"{metrics['precision']:.2%}")
c1.metric("Recall", f"{metrics['recall']:.2%}")
c2.metric("F1 Score", f"{metrics['f1']:.2%}")
st.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

st.write("**Confusion Matrix**")
st.caption("Rows = Actual class; columns = Predicted class.")
st.dataframe(pd.DataFrame(
    metrics["cm"],
    index=["Actual: Absence", "Actual: Presence"],
    columns=["Predicted: Absence", "Predicted: Presence"]
), use_container_width=True)

# 8. Feature importance
st.divider()
st.subheader("Feature Importance")
st.caption("Feature importance describes model split contribution, not medical causation.")
st.dataframe(importance, use_container_width=True)
st.caption("For academic demonstration only. Do not use this app for diagnosis or treatment decisions.")

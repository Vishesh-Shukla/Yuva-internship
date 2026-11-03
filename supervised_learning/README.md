# Heart Disease Prediction

Streamlit project using **Random Forest Classifier only**.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Workflow
1. Load and inspect dataset.
2. Check missing values, duplicate rows and target distribution.
3. Remove duplicates if present; verify missing values.
4. Encode target labels (`Absence = 0`, `Presence = 1`).
5. Separate features and target.
6. Split into training and testing sets using stratification.
7. Train Random Forest and evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC and confusion matrix.

The supplied dataset has no missing values or duplicate rows, so no imputation was required. Random Forest does not require feature scaling.

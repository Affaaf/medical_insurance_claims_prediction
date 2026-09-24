import joblib
import pandas as pd
import numpy as np

model = joblib.load("ml_models/xgboost_claim_model.pkl")
encoders = joblib.load("ml_models/encoders.pkl")
scaler = joblib.load("ml_models/scaler.pkl")


def predict_claim(new_claim: dict):

    new_df = pd.DataFrame([{k: v for k, v in new_claim}])
    cols_to_remove = ["Paid1", "Proc1_Adj"]
    new_df = new_df.drop(columns=[col for col in cols_to_remove if col in new_df.columns])

    expected_cols = ['InsuID', 'InsuPlanID', 'Diag1', 'Diag2', 'Diag3', 'Diag4', 'Diag5',
                    'Diag6', 'Diag7', 'Diag8', 'Diag9', 'Diag10', 'Diag11', 'Diag12',
                    'Proc1', 'Modifier1', 'Proc2', 'Modifier2', 'Paid2', 'Proc2_Adj',
                    'Proc3', 'Modifier3', 'Paid3', 'Proc3_Adj', 'Proc4', 'Modifier4',
                    'Paid4', 'Proc4_Adj', 'Proc5', 'Modifier5', 'Paid5', 'Proc5_Adj',
                    'Proc6', 'Modifier6', 'Paid6', 'Proc6_Adj']

    paid_cols = ["Paid2", "Paid3", "Paid4", "Paid5", "Paid6"]

    for col in expected_cols:
        if col not in new_df.columns:
            if col in paid_cols:
                new_df[col] = 0
            else:
                new_df[col] = ""

    # ---- Encode categorical columns safely ----
    categorical_cols = list(encoders.keys())

    for col in categorical_cols:
        new_df[col] = new_df[col].replace("", "missing").astype(str)

        # Ensure 'missing' exists in encoder classes
        if 'missing' not in encoders[col].classes_:
            encoders[col].classes_ = np.append(encoders[col].classes_, 'missing')

        # Safe encoding: unseen values map to 'missing'
        def safe_encode(value, le):
            return le.transform([value])[0] if value in le.classes_ else le.transform(['missing'])[0]

        new_df[col] = new_df[col].apply(lambda x: safe_encode(x, encoders[col]))

    # ---- Apply same scaling ----
    new_df[categorical_cols] = scaler.transform(new_df[categorical_cols])

    # ---- Ensure all columns are numeric ----
    new_df = new_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # ---- Align columns exactly as expected ----
    new_df = new_df[expected_cols]

    prediction = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "confidence": float(probability[0] if prediction[0] == 1 else 1 - probability[0])

    }

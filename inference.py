"""SageMaker inference entry point.

Generic sklearn Pipeline loader. Works for any pipeline saved as model.joblib,
regardless of which classifier is inside.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

# Target kelas klasifikasi risiko
CLASS_NAMES = ["High", "Moderate", "Low"]

# Mendaftarkan SEMUA kolom fitur sesuai dataset mentah (Kecuali Family_ID dan target)
FEATURE_NAMES = [
    "Father_Age",
    "Mother_Age",
    "Father_Height_cm",
    "Mother_Height_cm",
    "Father_Blood_Group",
    "Mother_Blood_Group",
    "Father_Eye_Color",
    "Mother_Eye_Color",
    "Father_Hair_Color",
    "Mother_Hair_Color",
    "Father_Skin_Tone",
    "Mother_Skin_Tone",
    "Family_Disease_History",
    "Predicted_Child_Blood_Group",
    "Child_Gender"
]


def model_fn(model_dir):

    return joblib.load(
        os.path.join(model_dir, "model_parental_logreg.pkl")
    )


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming request body into a DataFrame."""
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = [
            [x.strip() for x in line.split(",")]
            for line in request_body.strip().splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    # Pastikan data kosong/NaN pada data riwayat penyakit diisi string 'None' agar tidak error
    if "Family_Disease_History" in input_data.columns:
        input_data["Family_Disease_History"] = input_data["Family_Disease_History"].fillna("None")

    probs = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize the prediction dict for the response body."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")

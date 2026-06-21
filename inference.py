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
# PERBAIKAN: Pastikan nama kolom "Predicted_Child_Blood_Group" ini sesuai dengan 
# nama kolom yang digunakan saat training pipeline scikit-learn kamu!
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
    "Child_Blood_Group", 
    "Child_Gender"
]


def model_fn(model_dir: str):
    """Load the pickled sklearn Pipeline from model.joblib."""
    return joblib.load(os.path.join(model_dir, "model_parental.joblib"))


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming request body into a DataFrame and enforce types."""
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        df = pd.DataFrame(instances, columns=FEATURE_NAMES)

    elif request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = [
            [x.strip() for x in line.split(",")]
            for line in request_body.strip().splitlines()
            if line.strip()
        ]
        df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # =========================================================================
    # KUNCI UTAMA: Paksa kolom numerik menjadi float/int agar saat masuk ke 
    # pipeline preprocessing tidak error (terutama jika input dikirim lewat CSV)
    # =========================================================================
    numeric_cols = ["Father_Age", "Mother_Age", "Father_Height_cm", "Mother_Height_cm"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Isi nilai kosong/NaN numerik dengan median default jika diperlukan
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    # Pastikan sisa kolom kategori berupa string murni
    string_cols = [col for col in FEATURE_NAMES if col not in numeric_cols]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    return df


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    
    # Menjalankan pipeline scikit-learn (otomatis memproses fungsi preprocess internal di dalamnya)
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
    if accept_content_type == JSON_CONTENT_TYPE or accept_content_type == "*/*":
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
"""
Streamlit UI for the Parental Health Risk classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "parental-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.title("Parental Genetics & Child Health Risk Predictor")
st.write("Masukkan seluruh data orang tua dan anak di bawah ini untuk memprediksi risiko kesehatan via SageMaker.")

# --- 1. SETUP USER INPUTS (MENGGUNAKAN SEMUA FEATURE) ---

st.subheader("📊 Data Fisik & Umur Orang Tua")
col1, col2 = st.columns(2)
with col1:
    father_age = st.slider("Umur Ayah (Tahun)", 20, 55, 37)
    father_height = st.slider("Tinggi Badan Ayah (cm)", 155.0, 198.0, 174.0)
with col2:
    mother_age = st.slider("Umur Ibu (Tahun)", 18, 50, 34)
    mother_height = st.slider("Tinggi Badan Ibu (cm)", 145.0, 185.0, 161.0)


st.subheader("🩸 Golongan Darah & Profil Anak")
col3, col4, col5 = st.columns(3)
father_blood_group = st.selectbox(
    "Golongan Darah Ayah",
    ['O+', 'A+', 'A-', 'B-', 'O-', 'AB+', 'B+', 'AB-'],
    key="father_blood"
)

mother_blood_group = st.selectbox(
    "Golongan Darah Ibu",
    ['B+', 'A+', 'AB-', 'O+', 'AB+', 'A-', 'O-', 'B-'],
    key="mother_blood"
)

child_blood_group = st.selectbox(
    "Golongan Darah Anak",
    ['O+', 'A+', 'A-', 'B-', 'O-', 'AB+', 'B+', 'AB-'],
    key="child_blood"
)

st.subheader("👁️🧬 Ciri Fisik Genetik")
col6, col7, col8 = st.columns(3)
with col6:
    father_eye_color = st.selectbox(
        "Warna Mata Ayah", 
        ['Green', 'Brown', 'Blue', 'Hazel', 'Gray']
    )
    mother_eye_color = st.selectbox(
        "Warna Mata Ibu", 
        ['Brown', 'Blue', 'Black', 'Hazel', 'Green', 'Gray']
    )
with col7:
    father_hair_color = st.selectbox(
        "Warna Rambut Ayah", 
        ['Black', 'Brown', 'Blonde', 'Red']
    )
    mother_hair_color = st.selectbox(
        "Warna Rambut Ibu", 
        ['Brown', 'Black', 'Blonde', 'Red']
    )
with col8:
    father_skin_tone = st.selectbox(
        "Warna Kulit Ayah", 
        ['Olive', 'Fair', 'Medium', 'Dark', 'Light']
    )
    mother_skin_tone = st.selectbox(
        "Warna Kulit Ibu", 
        ['Light', 'Medium', 'Olive', 'Fair', 'Dark']
    )


st.subheader("🏥 Riwayat Kesehatan Keluarga")
family_disease_history = st.selectbox(
    "Riwayat Penyakit Terakhir/Keluarga", 
    ['None', 'Multiple', 'Hypertension', 'Diabetes', 'Asthma', 'Heart Disease']
)




# --- 2. PROSES PREDIKSI ---

if st.button("Predict", type="primary"):
    # PENTING: Urutan variabel di dalam list 'features' ini HARUS SAMA PERSIS 
    # dengan urutan FEATURE_NAMES yang terdaftar pada file inference.py di SageMaker!
    features = [
        father_age,
        mother_age,
        father_height,
        mother_height,
        father_blood_group,
        mother_blood_group,
        father_eye_color,
        mother_eye_color,
        father_hair_color,
        mother_hair_color,
        father_skin_tone,
        mother_skin_tone,
        family_disease_history,
        child_blood_group,
        child_gender
    ]
    
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        # Menampilkan Hasil Prediksi Risiko Kesehatan (High / Moderate / Low)
        label = result["labels"][0]
        st.success(f"Predicted Health Risk: **{label}**")
        
        # Menampilkan Probabilitas Tiap Kelas (Jika dikembangkan oleh endpoint)
        if "probabilities" in result:
            probs = result["probabilities"][0]
            st.write("Class probabilities:")
            
            classes = ["High", "Moderate", "Low"] 
            if len(probs) == len(classes):
                chart_data = {classes[i]: probs[i] for i in range(len(classes))}
                st.bar_chart(chart_data)
            else:
                st.bar_chart({"probability": probs})

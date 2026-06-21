import json
import os
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "iris-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features: list[float]) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),)
    return json.loads(response["Body"].read().decode("utf-8"))

st.title("Iris Predictor")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict", type="primary"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found")
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]
        st.success(f"Predicted quality: **{label}**")


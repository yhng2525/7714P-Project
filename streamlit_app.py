import joblib
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import create_sliding_windows, TimeSeriesScaler  # make sure TimeSeriesScaler is in utils
from constants import local_joblib_model_dir

# ===== Config =====
class_names = ['Brush_teeth', 'Climb_stairs', 'Comb_hair', 'Descend_stairs', 'Drink_glass',
 'Eat_meat', 'Eat_soup', 'Getup_bed', 'Liedown_bed', 'Pour_water',
 'Sitdown_chair', 'Standup_chair', 'Use_telephone', 'Walk']

window_size = 192
step_size = 48

# ===== Load 1D-CNN Model =====
cnn_model = load_model(f"{local_joblib_model_dir}/best_1dcnn_model_0813_1251.h5")
# ===== Load pre-fitted scaler pipeline =====
scaler_path = f"{local_joblib_model_dir}/cnn_preprocess_pipeline.joblib"
scaling_pipeline = joblib.load(scaler_path)


# ===== Streamlit UI =====
st.title("HMP Classification with Accelerometer Data (1D-CNN)")

st.write(f"Window Size: {window_size}, Step Size: {step_size}")

# File uploader
uploaded_file = st.file_uploader("Upload Accelerometer Data (.txt)", type=["txt"])

if uploaded_file is not None:
    # Read TXT file, space-separated, no header
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=["x", "y", "z"])
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    # Convert to numpy array
    data_array = df[["x", "y", "z"]].to_numpy()

    # Dummy labels for sliding windows
    dummy_labels = np.zeros(len(data_array))

    # Create sliding windows
    X_windows, _ = create_sliding_windows(data_array, dummy_labels, window_size, step_size, label_strategy="last")
    st.write(f"Created {X_windows.shape[0]} windows, each with shape {X_windows.shape[1:]}")

    # Apply scaler pipeline (transform only, no fitting!)
    X_scaled = scaling_pipeline.transform(X_windows)
    
    # Predict button
    if st.button("Predict"):
        # CNN expects (samples, timesteps, features)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], window_size, 3)
        predictions = np.argmax(cnn_model.predict(X_cnn), axis=1)
        pred_names = [class_names[i] for i in predictions]

        st.success("Prediction Completed")
        st.write("Predicted classes for each window:")
        st.write(pred_names)

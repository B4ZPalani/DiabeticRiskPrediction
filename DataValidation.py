import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats

# Load your dataset
df = pd.read_csv("data/diabetic_prediction.csv")

# Check for missing values
st.subheader("Check for Missing Values")
if df.isnull().sum().any():
    st.write("Dataset contains missing values.")
    
    # Count missing values per column
    missing_counts = df.isnull().sum()
    
    # Display count of missing values per column
    st.write("Missing values count per column:")
    st.write(missing_counts[missing_counts > 0])  # Display only columns with missing values
else:
    st.write("No missing values detected.")

# Display current data types
st.subheader("Current data types:")
st.write(df.dtypes)

# Outlier Detection
st.subheader("Outlier Detection")

# Select a column for outlier detection (numeric columns only)
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
selected_column = st.selectbox("Select a column to check for outliers", numeric_columns)

# Outlier detection using IQR method
def detect_outliers_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Outlier detection using Z-score method
def detect_outliers_zscore(column):
    z_scores = np.abs(stats.zscore(df[column]))
    threshold = 3  # Common threshold is 3
    outliers = df[z_scores > threshold]
    return outliers

# Choose outlier detection method
method = st.radio("Select outlier detection method", ("IQR", "Z-score"))

# Display outliers
if method == "IQR":
    st.write(f"Outliers in column '{selected_column}' using IQR method:")
    iqr_outliers = detect_outliers_iqr(selected_column)
    st.write(iqr_outliers)
    
    st.write(f"Number of outliers detected: {iqr_outliers.shape[0]}")
else:
    st.write(f"Outliers in column '{selected_column}' using Z-score method:")
    zscore_outliers = detect_outliers_zscore(selected_column)
    st.write(zscore_outliers)

    st.write(f"Number of outliers detected: {zscore_outliers.shape[0]}")

# Option to remove outliers
if st.button("Remove Outliers"):
    if method == "IQR":
        df = df[~df.index.isin(iqr_outliers.index)]
        st.write("Outliers removed using IQR method.")
    else:
        df = df[~df.index.isin(zscore_outliers.index)]
        st.write("Outliers removed using Z-score method.")
    
    st.write("Dataset after removing outliers:")
    st.write(df)


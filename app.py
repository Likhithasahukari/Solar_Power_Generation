import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import os
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
st.set_page_config(page_title="Solar Power Generation", layout="wide")
st.title("Solar Power Generation")
def load_and_clean(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = (
        df.columns.str.replace('-', '_')
                  .str.replace(' ', '_')
                  .str.replace('(', '')
                  .str.replace(')', '')
    )
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(0)
    return df
def auto_find_dataset():
    paths = [
        "solarpowergeneration.csv",
        "solarpowergeneration (1).csv",
        "/mnt/data/solarpowergeneration.csv",
        "/mnt/data/solarpowergeneration (1).csv",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return load_and_clean(p), p
            except:
                pass
    return None, None
df, used_path = auto_find_dataset()
st.sidebar.header("Dataset Loader")
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
if uploaded:
    df = load_and_clean(uploaded.name)
    used_path = uploaded.name
if df is None:
    st.warning("Upload a dataset to continue.")
    st.stop()
st.success(f"Dataset Loaded: {used_path}")
st.write(df.head())
TARGET = "power_generated"
if TARGET not in df.columns:
    guess = get_close_matches(TARGET, df.columns, n=1, cutoff=0.4)
    if guess:
        TARGET = guess[0]
        st.info(f"Target auto-detected as: {TARGET}")
    else:
        st.error("Target column not found. Available columns:")
        st.write(df.columns.tolist())
        st.stop()
st.header("Exploratory Data Analysis")
with st.expander("Summary Statistics"):
    st.write(df.describe())
with st.expander("Missing Values"):
    st.write(df.isnull().sum())
with st.expander("Numeric Distributions"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.multiselect("Select Columns", num_cols, default=num_cols[:5])
    if selected:
        df[selected].hist(figsize=(12, 8))
        st.pyplot(plt.gcf())
with st.expander("Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm")
    st.pyplot(fig)
st.header("Feature Selection")
feature_candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
selected_features = st.multiselect("Choose Features", feature_candidates, default=feature_candidates)
X = df[selected_features]
y = df[TARGET]
test_size = st.sidebar.slider("Test Size", 0.05, 0.5, 0.2)
rand_state = st.sidebar.number_input("Random State", 0, 9999, 42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=rand_state)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(random_state=rand_state),
    "Random Forest": RandomForestRegressor(random_state=rand_state),
    "Gradient Boosting": GradientBoostingRegressor(random_state=rand_state),
}
st.header("Manual Model Training")
manual_choice = st.selectbox("Choose a Model", list(models.keys()))
manual_model = models[manual_choice]
manual_model.fit(X_train_s, y_train)
manual_preds = manual_model.predict(X_val_s)
rmse = np.sqrt(mean_squared_error(y_val, manual_preds))
st.subheader("Manual Model Performance")
st.write("R²:", r2_score(y_val, manual_preds))
st.write("MAE:", mean_absolute_error(y_val, manual_preds))
st.write("RMSE:", rmse)
with st.expander("Residual Plot"):
    fig, ax = plt.subplots()
    ax.scatter(y_val, manual_preds)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)
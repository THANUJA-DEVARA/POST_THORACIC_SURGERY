import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Thoracic Surgery Life Expectancy Predictor", layout="centered")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\manis\OneDrive\Desktop\LIFE\LIFE\Prediction-of-Life-Expectancy-Post-Thoracic-Surgery-Application-master\ThoraricSurgery.csv")
    le = LabelEncoder()

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop('Risk1Yr', axis=1)  # Replace 'Risk1Yr' if your target column name is different
    y = data['Risk1Yr']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, X.columns.tolist()

# App title
st.title("🫁 One year life expectancy post thoracic surgery using streamlit")
st.markdown("This app predicts whether a patient will survive 1 year after thoracic surgery using health data.")

# Load data and model
data = load_data()
model, feature_names = train_model(data)

# Sidebar input
st.sidebar.header("Enter Patient Details")

def user_input_features():
    inputs = {}
    for col in feature_names:
        if data[col].nunique() < 10:
            inputs[col] = st.sidebar.selectbox(f"{col}", sorted(data[col].unique()))
        else:
            inputs[col] = st.sidebar.slider(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    return pd.DataFrame([inputs])

input_df = user_input_features()

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result:")
    if prediction == 0:
        st.success(f"✅ The patient is likely to survive the first year after surgery. (Confidence: {prob[0]*100:.2f}%)")
    else:
        st.error(f"⚠️ The patient may not survive the first year. (Confidence: {prob[1]*100:.2f}%)")

# Show raw data
with st.expander("📊 View Dataset"):
    st.dataframe(data)

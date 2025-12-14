import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from streamlit_lottie import st_lottie
from pathlib import Path

# --- PATH CONFIGURATION ---
# IMPORTANT: Adjusting the path to account for the inner folder structure 
# (Automated-Fraud-Detection-System-main) during deployment.
# We assume the app is run from the root, but the files are in the subfolder.
# If you moved app.py to the root, change the subfolder name below to ''
SUBFOLDER_NAME = 'Automated-Fraud-Detection-System-main'
current_dir = Path(__file__).parent / SUBFOLDER_NAME

csv_path = current_dir / 'samp_online.csv'
model_path = current_dir / 'fraud.h5'

# ====================================================================
# --- CRITICAL PERFORMANCE FIX: CACHING THE MODEL ---
# The @st.cache_resource decorator ensures this large model is loaded 
# ONLY ONCE when the application starts, dramatically improving speed 
# on subsequent button clicks and reruns.
@st.cache_resource
def load_model_cached(path):
    # This block runs only on the very first execution
    with st.spinner('Loading Deep Learning Model... This may take a moment on the first run.'):
        # Load the Keras model
        model = tf.keras.models.load_model(path, compile=False)
        # Recompile the model (if necessary for prediction, though often predict() is enough)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load the model object immediately. It will be cached.
load_clf = load_model_cached(model_path)
# ====================================================================


st.title("Automated Fraud Detection System Web app")
st.write("""
This app will helps us to track what type of transactions lead to fraud. I collected a dataset from [Kaggle repositry](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)
,which contains historical information about fraudulent transactions which can be used to detect fraud in online payments.
""")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets8.lottiefiles.com/packages/lf20_yhTqG2.json"
lottie_hello = load_lottieurl(lottie_url)

with st.sidebar:
    st_lottie(lottie_hello, quality='high')

st.sidebar.title('Users Features Explanation')
st.sidebar.markdown("**step**: represents a unit of time where 1 step equals 1 hour")
st.sidebar.markdown("**type**: type of online transaction")
st.sidebar.markdown('**amount**: the amount of the transaction')
st.sidebar.markdown('**oldbalanceOrg**: balance before the transaction')
st.sidebar.markdown('**newbalanceOrig**: balance after the transaction')
st.sidebar.markdown('**oldbalanceDest**: initial balance of recipient before the transaction')
st.sidebar.markdown('**newbalanceDest**: the new balance of recipient after the transaction')

st.header('User Input Features')

def user_input_features():
    # Adjusted step range for better user experience, typically step goes up to 743
    step = st.number_input('Step (Time unit in hours)', 0, 744, value=1) 
    type = st.selectbox('Online Transaction Type', ("CASH IN", "CASH OUT", "DEBIT", "PAYMENT", "TRANSFER"))
    amount = st.number_input("Amount of the transaction", value=100.0)
    oldbalanceOrg = st.number_input("Old balance Origin", value=1000.0)
    newbalanceOrig = st.number_input("New balance Origin", value=900.0)
    oldbalanceDest = st.number_input("Old Balance Destination", value=500.0)
    newbalanceDest = st.number_input("New Balance Destination", value=600.0)
    
    data = {'step': step,
            'type': type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combines user input features with sample dataset
@st.cache_data
def load_and_preprocess_data(path, input_df):
    """Loads CSV, performs preprocessing, and returns the final dataframe."""
    try:
        # Load data
        fraud_raw = pd.read_csv(path)
        fraud = fraud_raw.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
        
        # Concatenate user input with data for consistent encoding
        df = pd.concat([input_df, fraud], axis=0)
        
        # Encoding of ordinal features
        encode = ['type']
        for col in encode:
            dummy = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummy], axis=1)
            del df[col]
            
        # Selects only the first row (the user input data)
        return df[:1] 
        
    except FileNotFoundError:
        st.error(f"Error: Could not find '{path}'. Please check your file paths.")
        st.stop()

# Load and preprocess the data once
df = load_and_preprocess_data(csv_path, input_df)

# Reads in saved classification model
if st.button("Predict"):
    try:
        # Apply model to make predictions using the cached model (load_clf)
        # We DO NOT load the model here, as it's already in memory.
        
        # Ensure all columns expected by the model are present in the input data
        # Common issue: Model trained on 5 dummy columns, but input only has 1
        
        y_probs = load_clf.predict(df)
        pred = tf.round(y_probs)
        pred = tf.cast(pred, tf.int32)

        st.markdown(
            """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 25px;
        }
        </style>
            """,
            unsafe_allow_html=True,
        )

        if pred == 0:
            col1, col2 = st.columns(2)
            col1.metric("Prediction", value="Transaction is not fraudulent")
            col2.metric("Confidence Level", value=f"{np.round(np.max(y_probs) * 100)}%")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Prediction", value="Transaction is fraudulent")
            col2.metric("Confidence Level", value=f"{np.round(np.max(y_probs) * 100)}%")
            
    except Exception as e:
        st.error(f"Error predicting: {e}") 
        st.write("Ensure the input features match the model's training features.")

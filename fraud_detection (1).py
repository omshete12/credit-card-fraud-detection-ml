import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. CONFIGURATION ---
MODEL_PATH = 'fraud_detection_pipeline.pkl'
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model(path):
    """Loads the serialized model/pipeline."""
    try:
        with open(path, 'rb') as file:
            pipeline = pickle.load(file)
        st.success(f"Pipeline loaded successfully! Best model: {pipeline.named_steps['classifier'].__class__.__name__}")
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. Please ensure it was created by 'analysis_model.ipynb'.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_model(MODEL_PATH)

# The model expects features in the order they were trained (V1...V28, Amount).
# We define the expected features list here, matching the dataset structure.
V_FEATURES = [f'V{i}' for i in range(1, 29)]
FEATURE_ORDER = V_FEATURES + ['Amount']


# --- 3. STREAMLIT UI ---
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

if pipeline:
    st.header("Enter Transaction Details")
    st.info("Input fields correspond to the anonymized features (V1-V28) and Transaction Amount.")

    # Dictionary to store user inputs
    all_inputs = {}
    
    # --- Input Fields for V1 to V28 ---
    st.subheader("Anonymized Features (V1-V28)")
    
    # Use columns for a compact layout (4 columns wide)
    cols = st.columns(4)
    
    # Create input fields for V1 to V28
    for i, feature_name in enumerate(V_FEATURES):
        col = cols[i % 4]
        with col:
            # Use number_input for floating-point values, setting a neutral default
            all_inputs[feature_name] = st.number_input(
                f'{feature_name}', 
                value=0.0, 
                step=0.001, 
                format="%.6f",
                key=f'input_{feature_name}',
                help=f"Input value for {feature_name}"
            )

    # --- Input Field for Amount ---
    st.subheader("Transaction Amount")
    
    all_inputs['Amount'] = st.number_input(
        'Amount', 
        value=50.0, # A typical non-fraud amount default
        min_value=0.0, 
        step=1.0, 
        format="%.2f",
        help="The dollar value of the transaction"
    )

    # --- PREDICTION BUTTON ---
    st.markdown("---")
    if st.button('Analyze Transaction for Fraud', use_container_width=True):
        
        # 1. Create a DataFrame from the inputs, ensuring correct column order
        try:
            input_data = pd.DataFrame([all_inputs], columns=FEATURE_ORDER)
            
            # 2. Make the prediction
            # The pipeline automatically handles scaling and prediction
            prediction = pipeline.predict(input_data)[0]
            prediction_proba = pipeline.predict_proba(input_data)[0]

            # 3. Display the result
            st.subheader("Prediction Result")
            
            # Probability of the positive class (Fraud = 1)
            fraud_probability = prediction_proba[1] 
            
            if prediction == 1:
                st.error(f"🔴 **FRAUDULENT TRANSACTION ALERT**")
                st.warning(f"Estimated Probability of Fraud: **{fraud_probability:.4f}** ({fraud_probability:.2%})")
                st.markdown("🚨 **The model classifies this transaction as high-risk.**")
            else:
                st.success(f"🟢 **VALID TRANSACTION**")
                st.info(f"Estimated Probability of Fraud: **{fraud_probability:.4f}** ({fraud_probability:.2%})")
                st.markdown("✅ **Transaction deemed safe based on current model parameters.**")

        except ValueError as e:
            st.error(f"Input Error: Please check all {len(FEATURE_ORDER)} input fields. {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

else:
    st.warning("Cannot run prediction until the model is loaded successfully. Check the file path and logs.")
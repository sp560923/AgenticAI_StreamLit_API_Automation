import streamlit as st
import pandas as pd
import json

from main import run_single_request  # ADD THIS: Import the function that has Allure logic
st.set_page_config(page_title="Agentic API Tester", page_icon="🤖", layout="wide")

st.title("🤖 Agentic API Testing Suite")
st.markdown("Enter details for a single microservice or upload a file for bulk testing.")

# --- SIDEBAR: MODE SELECTION ---
mode = st.sidebar.selectbox("Select Mode", ["Single Request", "Bulk Upload"])

if mode == "Single Request":
    st.subheader("Manual API Input")
    
    endpoint = st.text_input("Endpoint URL", "https://api.restful-api.dev/objects")
    method = st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
    headers = st.text_area("Headers (JSON string)", '{"Content-type": "application/json"}')
    body = st.text_area("Body (JSON string)", '{}')

    if st.button("Run Agentic Test"):
        raw_input = f"Endpoint: {endpoint}\nMethod: {method}\nHeaders: {headers}\nBody: {body}"
        
        with st.spinner("Agents are working..."):
            try:
                result = run_single_request(raw_input)
                
                st.success("Test Complete!")
                st.markdown("### Agent Report")
                st.markdown(result.raw)
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.subheader("Bulk Microservice Testing")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
    
    if uploaded_file and st.button("Start Bulk Execution"):
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            
        st.write(f"Found {len(df)} microservices. Starting sequential run...")
        progress_bar = st.progress(0)
        
        for i, row in df.iterrows():
            # --- SUGGESTED CHANGE: DATA CLEANING FOR BULK ---
            # Extract values safely to avoid None/NaN values causing Pydantic errors
            url = str(row.get('url', '')).strip()
            method = str(row.get('method', 'GET')).strip()
            
            # Handle Headers: ensure it's a valid string or empty dict string
            raw_headers = row.get('headers', '{}')
            headers_str = str(raw_headers) if pd.notna(raw_headers) else '{}'
            
            # Handle Body: ensure it's a valid string or empty dict string
            raw_body = row.get('body', '{}')
            body_str = str(raw_body) if pd.notna(raw_body) else '{}'

            service_info = f"URL: {url}\nMethod: {method}\nHeaders: {headers_str}\nBody: {body_str}"
            
            with st.expander(f"Testing Service {i+1}: {url}"):
                try:
                    # Calling the main controller which handles the crew and reporting
                    result = run_single_request(service_info)
                    st.markdown(result.raw)
                except Exception as e:
                    st.error(f"Failed: {e}")
            
            progress_bar.progress((i + 1) / len(df))
        
        st.balloons()
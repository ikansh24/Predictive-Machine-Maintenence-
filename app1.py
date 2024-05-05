import streamlit as st
import pandas as pd
import pickle
from io import BytesIO

# Initialize session state for login status if not already set
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Load the models and LabelEncoder
with open('model_failure.pkl', 'rb') as f:
    model_failure = pickle.load(f)
with open('model_critical.pkl', 'rb') as f:
    model_critical = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def check_login(username, password):
    # Placeholder check for username and password
    return username == "admin" and password == "password"

def login_form():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.sidebar.error("Incorrect username or password")

def logout_user():
    st.session_state.logged_in = False
    st.experimental_rerun()

# If not logged in, show login; else show the main app
if not st.session_state.logged_in:
    login_form()
else:
    st.title('Machine Failure Prediction Tool')

    # Logout button
    if st.sidebar.button("Logout"):
        logout_user()

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.write(data.head())

            # Making predictions
            data['Failure'] = model_failure.predict(data[['Temperature', 'Vibration', 'Pressure', 'Humidity', 'Operational_Hours']])
            critical_metrics = model_critical.predict(data[['Temperature', 'Vibration', 'Pressure', 'Humidity', 'Operational_Hours']])
            data['Critical Metric'] = label_encoder.inverse_transform(critical_metrics)

            data = data[(data['Failure'] != 0) & (data['Critical Metric'] != 'None')]

            # Group data by 'Critical Metric'
            grouped_data = data.groupby('Critical Metric')

            # Concatenate all groups into one DataFrame with a group identifier
            all_groups = pd.concat([group.assign(Group=name) for name, group in grouped_data])

            # Show the grouped data
            st.write("Filtered and Grouped Predictions:")
            st.dataframe(all_groups[['Machine_ID', 'Failure', 'Critical Metric', 'Group']])

            # Download link for the grouped data as CSV
            output = BytesIO()
            all_groups.to_csv(output, index=False)
            output.seek(0)
            st.download_button(label="Download Grouped Data as CSV",
                               data=output,
                               file_name='grouped_predictions.csv',
                               mime='text/csv')
        else:
            st.error("Invalid data file. Please upload a CSV.")

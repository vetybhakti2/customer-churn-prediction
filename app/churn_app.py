import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import classification_report
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model

# Load data
data = load_data()

# Preprocess data
X, y, customer_ids = preprocess_data(data)

# Train model
model, accuracy, report = train_model(X, y)

# Streamlit app
st.title('Customer Churn Prediction')

# Sidebar for navigation
st.sidebar.header('Navigation')
option = st.sidebar.selectbox(
    'Choose a page:',
    ['Home', 'EDA', 'Predict Churn']
)

# Home page
if option == 'Home':
    st.image('/Users/vetybhakti/Documents/Vety/customer-churn-prediction/app/image.png', width=600)
    st.write('Author: Vety Bhakti Lestari')
    st.write('Welcome to the Customer Churn Prediction App!')
    st.write('Use the sidebar to navigate to different sections.')

    # Show raw data
    if st.checkbox('Show raw data'):
        st.write(data)

# EDA page with Plotly
elif option == 'EDA':
    st.header('Exploratory Data Analysis (EDA)')

    # Visualize the distribution of churn
    churn_dist = data['Churn'].value_counts().reset_index()
    churn_dist.columns = ['Churn', 'Count']
    fig = px.pie(churn_dist, names='Churn', values='Count', title="Churn Distribution")
    st.plotly_chart(fig)

    # Visualize distribution of a specific feature, grouped by churn
    feature = st.selectbox('Select feature for visualization:', data.columns)
    if feature:
        # Group the data by 'Churn' and plot the histogram
        fig = px.histogram(data, x=feature, color='Churn', title=f'{feature} Distribution Grouped by Churn', 
                           barmode='overlay', nbins=20)
        st.plotly_chart(fig)

# Predict Churn page
elif option == 'Predict Churn':
    st.header('Predict Customer Churn')

    # Upload new data
    uploaded_file = st.file_uploader("Upload new data for prediction", type=["csv"])
    if uploaded_file is not None:
        # Load the uploaded data
        new_data = pd.read_csv(uploaded_file)

        # Preprocess new data (assuming you have the same preprocessing function)
        X_new, y_new, customer_ids_new = preprocess_data(new_data)

        # Predict churn for the new data
        y_pred_new = model.predict(X_new)

        # Create a DataFrame with customerID and predictions
        results_new = pd.DataFrame({
            'customerID': customer_ids_new,
            'Churn_Prediction': y_pred_new
        })

        # Map predictions to 'Yes' or 'No' for better readability
        results_new['Churn_Prediction'] = results_new['Churn_Prediction'].map({1: 'Yes', 0: 'No'})

        # Display results for the new data
        st.write('Predictions for uploaded data:')
        st.write(results_new)

        # Option to download the new predictions
        if st.button('Download New Predictions as CSV'):
            results_new.to_csv('new_churn_predictions.csv', index=False)
            st.write('Predictions downloaded as `new_churn_predictions.csv`.')

    else:
        # Make predictions for existing data if no new file is uploaded
        y_pred = model.predict(X)

        # Display model accuracy with a nice gauge chart
        st.write(f'Model Accuracy: {accuracy:.2f}')
        accuracy_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            title={'text': "Model Accuracy"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "lightgreen"}, 'steps': [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ]}
        ))
        st.plotly_chart(accuracy_gauge)

        # Generate classification report and display it
        st.write("### Classification Report:")
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(axis=None, gmap=report_df.values, cmap='YlGnBu'))

        # Create a DataFrame with customerID and predictions
        results = pd.DataFrame({
            'customerID': customer_ids,
            'Churn_Prediction': y_pred
        })
        results['Churn_Prediction'] = results['Churn_Prediction'].map({1: 'Yes', 0: 'No'})

        # Display results
        st.write('Predictions for existing data:')
        st.write(results)
# Customer Churn Prediction

This project aims to predict customer churn based on various customer-related features using machine learning. It leverages XGBoost and Streamlit for data visualization and model deployment.

## Features
- **Churn Prediction**: Predict whether a customer is likely to churn (leave) or stay.
- **Exploratory Data Analysis (EDA)**: Visualizations to explore the distribution of data and churn rates.
- **Model Evaluation**: Model accuracy and detailed classification report.
- **Download Predictions**: Option to download churn prediction results as a CSV file.
- **Interactive Interface**: Easy-to-use web app built using Streamlit.

## Technologies Used
- **Python 3.12**
- **Streamlit**: For creating the web interface.
- **XGBoost**: For the churn prediction model.
- **Scikit-learn**: For model training and evaluation.
- **Pandas**: For data manipulation and preprocessing.
- **Plotly**: For data visualization.
- **Joblib**: For saving and loading the trained model.

## Prerequisites
Before running the project, make sure you have the following installed:

- Python 3.12+
- poetry (Python package manager)

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/vetybhakti2/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. **Install dependencies**:  

    ```bash
    poetry install
    ```

3. **Prepare the data**:
    Ensure that you have the customer churn dataset available. The project assumes the dataset is located in the `data` folder or is accessible through a specified path in the code.

4. **Run the app**:
    You can start the app by running the following command:

    ```bash
    streamlit run app/churn_app.py
    ```

    This will open the app in your web browser.

## How to Use the App

1. **Home Page**:
   - Displays the app's title and provides navigation options.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizes the churn distribution and allows you to explore features using histograms.

3. **Predict Churn**:
   - Display model accuracy and classification report.
   - Make predictions for customer churn.
   - Download predictions as a CSV file.

## Model Training

The model is trained using the XGBoost algorithm, and hyperparameters can be fine-tuned using GridSearchCV. The model is then saved using Joblib for future use.

### Train the Model
To train the model manually, run the following code snippet:

```python
from src.train_model import train_model

# Load and preprocess data
X, y, customer_ids = load_data()

# Train the model
model, accuracy, report = train_model(X, y)

# Save the model
joblib.dump(model, 'churn_model.pkl')

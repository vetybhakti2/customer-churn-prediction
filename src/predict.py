import pandas as pd
import joblib
from src.data_preprocessing import preprocess_data

def load_model(model_path):
    """
    Load the trained model from a file.
    """
    model = joblib.load(model_path)
    return model

def predict_churn(input_data, model):
    """
    Make predictions using the trained model.
    """
    # Preprocess the input data
    X, _, customer_ids = preprocess_data(input_data)

    # Make predictions
    predictions = model.predict(X)

    # Create a DataFrame with customerID and predictions
    results = pd.DataFrame({
        'customerID': customer_ids,
        'Churn_Prediction': predictions
    })

    # Map predictions to 'Yes' or 'No' for better readability
    results['Churn_Prediction'] = results['Churn_Prediction'].map({1: 'Yes', 0: 'No'})

    return results

# # Example usage
# if __name__ == "__main__":
#     # Load the trained model
#     model = load_model('/Users/vetybhakti/Documents/Vety/customer-churn-prediction/models/churn_model.pkl')

#     # Load input data (e.g., from a CSV file)
#     input_data = pd.read_csv('../data/new_customers.csv')  # Replace with your input data file

#     # Make predictions
#     predictions = predict_churn(input_data, model)

#     # Save or display the results
#     print(predictions)
#     predictions.to_csv('../data/predictions.csv', index=False)  # Save predictions to a CSV file
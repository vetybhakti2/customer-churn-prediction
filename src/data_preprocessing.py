import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    return pd.read_csv('/Users/vetybhakti/Documents/Vety/customer-churn-prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

def preprocess_data(data):
    # 1. Simpan customerID secara terpisah
    customer_ids = data['customerID']

    # 2. Drop customerID dari dataset
    data = data.drop(['customerID'], axis=1)

    # 3. Convert TotalCharges to numeric, handling errors
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # 4. Fill missing values in TotalCharges with the median
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # 5. Convert categorical variables to numeric using Label Encoding
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Churn':  # Exclude target column
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # 6. Convert target column 'Churn' to binary (1 for 'Yes', 0 for 'No')
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # 7. Separate features (X) and target (y)
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # 8. Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y, customer_ids
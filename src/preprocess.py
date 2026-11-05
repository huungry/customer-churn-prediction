from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def preprocess_data():

    df = pd.read_csv(DATA_PATH)

    #Change object to float for TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    #Fill missing values in TotalCharges to 0 because their tenure is 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    #Encode target value with LabelEncoder
    churn_encoder = LabelEncoder()
    y = df['Churn']

    #One hot encode categorical values & scale numerical values
    cols = df.select_dtypes(include=['object', 'str']).columns
    cols = [c for c in cols if (c!='customerID') & (c!='Churn')]
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cols_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    transformer = ColumnTransformer(transformers=[('cols_encoder', cols_encoder, cols), ('numerical_cols', StandardScaler(), numerical_cols)], verbose_feature_names_out=False)

    X = df.drop(columns=['customerID', 'Churn'])

    #Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    #Fit and transform data
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    y_train = churn_encoder.fit_transform(y_train)
    y_test = churn_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test



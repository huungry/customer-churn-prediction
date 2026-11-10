from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def fix_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def encode_numerical_columns(df):
    num_encoder = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = numerical_columns.drop('tenure')
    return num_encoder, numerical_columns

def encode_categorical_columns(df):
    cols = df.select_dtypes(include=['object', 'str']).columns
    cols = cols.drop(['customerID', 'Churn'])
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    return cat_encoder, cols

def encode_churn_column(df):
    churn_encoder = LabelEncoder()
    y = df['Churn']
    return churn_encoder, y

def preprocess_data():
    df = load_data()
    df = fix_data(df)

    num_encoder, num_cols = encode_numerical_columns(df)
    cat_encoder, cat_cols = encode_categorical_columns(df)
    churn_encoder, y = encode_churn_column(df)

    #One hot encode categorical values & scale numerical values
    transformer = ColumnTransformer(transformers=[('cols_encoder', cat_encoder, cat_cols), ('numerical_cols', num_encoder, num_cols)], verbose_feature_names_out=False)

    X = df.drop(columns=['customerID', 'Churn'])

    #Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Fit and transform data
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    y_train = churn_encoder.fit_transform(y_train)
    y_test = churn_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data()

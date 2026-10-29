import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Change object to float for TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


#One hot encode categorical values
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cols = df.select_dtypes(include=['str']).columns
cols = [c for c in cols if c!='customerID']
transformer = ColumnTransformer(transformers=[('encoder', encoder, cols)], remainder='passthrough', verbose_feature_names_out=False)
num_encode = transformer.fit_transform(df)

data = pd.DataFrame(num_encode, columns=transformer.get_feature_names_out(df.columns))
data = data.to_csv('../data/processed/processed_data.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('Dataset 2.csv')

# Keep only the necessary columns
df = df[['how much paid in last month.1', 'region', 'occupation (grouped)']]

# Replace 'don't know' with NaN and 'did not work in last month' with 0
df['how much paid in last month.1'] = df['how much paid in last month.1'].replace({'don\'t know': np.nan, 'did not work in last month': 0})

# Convert to numeric, setting failed conversions to NaN
df['how much paid in last month.1'] = pd.to_numeric(df['how much paid in last month.1'], errors='coerce')

# Impute missing income values with the median
df['how much paid in last month.1'] = df['how much paid in last month.1'].fillna(df['how much paid in last month.1'].median())

# Encode categorical features
object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Train-test Split
X = df.drop(columns='how much paid in last month.1')
y = df['how much paid in last month.1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = MinMaxScaler()
X_train_rescaled = scaler.fit_transform(X_train)
X_test_rescaled = scaler.transform(X_test)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)

# Train the model
rf_regressor.fit(X_train_rescaled, y_train)

# Predict on the test set
y_pred_rf = rf_regressor.predict(X_test_rescaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regressor MSE: {mse_rf}")
print(f"Random Forest Regressor R2 Score: {r2_rf}")

# Save the trained model
joblib.dump(rf_regressor, 'random_forest_regressor.pkl')

# Save the label encoders
with open('label_encoders.pkl', 'wb') as file:
    joblib.dump(label_encoders, file)

# Save the fitted scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)

# Save the feature names
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as file:
    joblib.dump(feature_names, file)

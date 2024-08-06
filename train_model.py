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

# Drop unnecessary columns
df = df.drop(columns=[
    'highest educational level', 'highest year of education', 'time to get to water source',
    'household has: refrigerator', 'religion','husband/partner\'s education level','ethnicity',
    'education in single years', 'household has: telephone (land-line)','educational level',
    'highest year of education (at level in mv106)', 'religion.1','partner education',
    'relationship to household head.1', 'sex of household head.1', 'age of household head.1',
    'literacy.1', 'owns a mobile telephone.1', 'last 12 months use mobile telephone for financial transactions.1',
    'is respondent\'s mobile phone a smart phone.1', 'has an account in a bank or other financial institution.1',
    'use of internet.1', 'frequency of using internet last month.1', 'self reported health status.1',
    'wealth index combined.1', 'husband/partner\'s total number of years of education',
    'justifies domestic violence: refuses to cook', 'respondent education.1','how much paid in last month',
    'occupation','respondent\'s occupation','case identification','wealth index for urban/rural','husband/partner\'s occupation'
])

# Replace 'don't know' with NaN and 'did not work in last month' with 0
df['how much paid in last month.1'] = df['how much paid in last month.1'].replace({'don\'t know': np.nan, 'did not work in last month': 0})

# Convert to numeric, setting failed conversions to NaN
df['how much paid in last month.1'] = pd.to_numeric(df['how much paid in last month.1'], errors='coerce')

# Impute missing income values with the median
df['how much paid in last month.1'] = df['how much paid in last month.1'].fillna(df['how much paid in last month.1'].median())

# Replace ".a" with "Unknown" in categorical columns
df['husband/partner\'s occupation (grouped)'] = df['husband/partner\'s occupation (grouped)'].replace('.a', 'Unknown')
df['type of cooking fuel'] = df['type of cooking fuel'].replace(['17', '15'], 'unknown')
df['occupation (grouped)'] = df['occupation (grouped)'].replace('.a', 'Unknown')
df['respondent\'s occupation (grouped)'] = df['respondent\'s occupation (grouped)'].replace('.a', 'Unknown')

# Fill missing values in 'is respondent's mobile phone a smart phone' with mode
df['is respondent\'s mobile phone a smart phone'] = df['is respondent\'s mobile phone a smart phone'].fillna(df['is respondent\'s mobile phone a smart phone'].mode()[0])

# Function to cap outliers using the IQR method
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Apply the function to each numeric column
numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    df = cap_outliers(df, column)

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

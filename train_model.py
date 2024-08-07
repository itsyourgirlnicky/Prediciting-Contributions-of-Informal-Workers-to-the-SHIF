import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('Dataset 2.csv')

# Keep only the necessary columns
df = df[['how much paid in last month.1', 'region', 'occupation (grouped)']]

# Checking for duplicate rows
duplicate_rows = df.duplicated().sum()

# Drop duplicate rows
df = df.drop_duplicates()

# Checking and handling unique values in 'how much paid in last month.1'
df['how much paid in last month.1'] = df['how much paid in last month.1'].replace({'don\'t know': np.nan, 'did not work in last month': 0})

# Convert to numeric, setting failed conversions to NaN
df['how much paid in last month.1'] = pd.to_numeric(df['how much paid in last month.1'], errors='coerce')

# Checking and handling unique values in 'occupation (grouped)'
df['occupation (grouped)'] = df['occupation (grouped)'].replace('.a', 'Unknown')

# Handling missing values
df['how much paid in last month.1'] = df['how much paid in last month.1'].fillna(df['how much paid in last month.1'].median())

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

# Feature Engineering
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

# Standardization
scaler = StandardScaler()
X_train_rescaled = scaler.fit_transform(X_train_rescaled)
X_test_rescaled = scaler.transform(X_test_rescaled)

# Cross Validation using Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
cv_scores = cross_val_score(clf, X_train_rescaled, y_train, cv=5)

# Dimensional Reduction using PCA
n_components = 2  
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_rescaled)
X_test_pca = pca.transform(X_test_rescaled)

# Regularization with Ridge Classifier
ridge = RidgeClassifier()
ridge_cv_scores = cross_val_score(ridge, X_train_pca, y_train, cv=5)

# Pruning with Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gbc_cv_scores = cross_val_score(gbc, X_train_pca, y_train, cv=5)

# Hyperparameter tuning for Random Forest Regressor using GridSearchCV with the best parameters 
param_grid = {
    'n_estimators': [300],
    'max_depth': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [4]
}

rf_regressor = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_rescaled, y_train)


# Train the model with best parameters
best_rf_regressor = grid_search.best_estimator_
best_rf_regressor.fit(X_train_rescaled, y_train)

# Predict on the test set
y_pred_rf = best_rf_regressor.predict(X_test_rescaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regressor MSE: {mse_rf}")
print(f"Random Forest Regressor R2 Score: {r2_rf}")

# Save the trained model
joblib.dump(best_rf_regressor, 'random_forest_regressor.pkl')

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

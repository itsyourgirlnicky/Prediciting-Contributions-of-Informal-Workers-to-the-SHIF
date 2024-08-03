import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the saved preprocessed datasets
X_train_resampled = pd.read_csv('X_train_resampled.csv')
X_test_rescaled = pd.read_csv('X_test_rescaled.csv')
y_train_resampled = pd.read_csv('y_train_resampled.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Initialize the Random Forest model with the best parameters
rf_clf = RandomForestClassifier(
    bootstrap=False, 
    max_depth=20, 
    min_samples_leaf=1, 
    min_samples_split=2, 
    n_estimators=200
)

# Train the model
rf_clf.fit(X_train_resampled, y_train_resampled)

# Save the trained model with the current scikit-learn version
joblib.dump(rf_clf, 'random_forest_model.pkl')

# Predict on the test set
y_pred_rf = rf_clf.predict(X_test_rescaled)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Classifier Accuracy: {accuracy_rf}")
print("Classification Report for Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test, y_pred_rf))

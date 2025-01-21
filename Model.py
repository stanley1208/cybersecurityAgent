from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("Phising_Training_Dataset.csv")

# Drop the 'key' column as it's not a feature
data = data.drop(columns=["key"])

# Define target and features
target_column = "Statistical_report"  # Last column in the dataset
y = data[target_column]  # Target variable
X = data.drop(columns=[target_column])  # Features

# Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, "phishing_detector_model.pkl")
joblib.dump(scaler, "scaler.pkl")


print(data.head())  # View processed data
print(X.columns)  # Ensure features are correctly extracted
print(X_train.shape, X_test.shape)  # Check data split

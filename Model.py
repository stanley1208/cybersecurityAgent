from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("Phising_Training_Dataset.csv")

# Drop 'key' column (not needed for training)
df.drop(columns=['key'], inplace=True)

# Convert -1 labels to 0 (Ensures consistency with Flask app)
df["Statistical_report"] = df["Statistical_report"].replace(-1, 0)

# Features (X) and Labels (y)
X = df.drop(columns=["Statistical_report", "Result"])  # Drop 'Result' if it exists

y = df["Statistical_report"]  # Labels

print("Training Features:", list(X.columns))
print("Total Features in Training:", len(X.columns))


# Normalize numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Apply SMOTE if needed (to balance classes)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Define model & hyperparameters
xgb_model = XGBClassifier()

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(xgb_model, params, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model & scaler for Flask app
joblib.dump(best_model, "phishing_detector_model.pkl")
joblib.dump(scaler, "scaler.pkl")

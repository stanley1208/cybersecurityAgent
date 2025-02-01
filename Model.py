from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("Phising_Training_Dataset.csv")


# Remove non-feature columns
if "Key" in data.columns:
    data=data.drop(column=["Key"])


# Define target & features
target_column="Statistical_report"
y=data[target_column].replace(-1,0) # Convert -1 to 0 for compatibility
X=data.drop(columns=[target_column,"Result"]) # Remove label columns


# Normalize numerical features
scaler=StandardScaler()
X_scaled=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

# Split dataset
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)


# Define model & hyperparameters
xgb_model=XGBClassifier()

params={
    'n_estimators':[100,200],
    'max_depth':[3,6],
    'learning_rate':[0.01,0.1],
    'colsample_bytree':[0.8,1.0]
}

grid_search=GridSearchCV(xgb_model,params,cv=3,scoring="accuracy",verbose=2)
grid_search.fit(X_train,y_train)

# Best model
best_model=grid_search.best_estimator_

# Evaluate
y_pred=best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model & scaler
joblib.dump(best_model,"phishing_detector_model.pkl")
joblib.dump(scaler,"scaler.pkl")


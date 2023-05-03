import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("kickstarter_data_with_features.csv", low_memory=False)

# Data preprocessing
non_useful_columns = ['friends', 'is_starred', 'is_backing', 'permissions', 'photo', 'name', 'blurb', 'slug', 'currency_symbol', 'creator', 'location', 'profile', 'urls', 'source_url']
df.drop(non_useful_columns, axis=1, inplace=True)

imputer = SimpleImputer(strategy='mean')
df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))

date_columns = ['deadline', 'state_changed_at', 'created_at', 'launched_at']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day

df.drop(date_columns, axis=1, inplace=True)

label_encoder = LabelEncoder()
categorical_columns = ["category", "currency", "country"]
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

df["campaign_length"] = (df["deadline_year"] - df["launched_at_year"]) * 365 + (df["deadline_month"] - df["launched_at_month"]) * 30 + (df["deadline_day"] - df["launched_at_day"])
df.drop(["launched_at_year", "deadline_year", "deadline_month", "deadline_day", "launched_at_month", "launched_at_day"], axis=1, inplace=True)

# Data splitting
X = df.drop("state", axis=1)
y = df["state"]

# Encode the target variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
X_train = pd.DataFrame(X_train, columns=X.select_dtypes(include=[np.number]).columns)
X_test = scaler.transform(X_test.select_dtypes(include=[np.number]))
X_test = pd.DataFrame(X_test, columns=X.select_dtypes(include=[np.number]).columns)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model training with XGBoost
clf = XGBClassifier(random_state=42, n_jobs=-1)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 10],
    'n_estimators': [50, 100, 200],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best parameters from GridSearchCV:", best_params)

# Train the XGBoost classifier with the best parameters
best_clf = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    n_estimators=best_params["n_estimators"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"]
)
best_clf.fit(X_train, y_train)

# Predictions
y_pred_train = best_clf.predict(X_train)
y_pred_test = best_clf.predict(X_test)

# Model performance
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

print("\nTraining Classification Report:\n", classification_report(y_train, y_pred_train))
print("Test Classification Report:\n", classification_report(y_test, y_pred_test))

print("\nTraining Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

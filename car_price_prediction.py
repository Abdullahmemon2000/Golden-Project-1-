import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the dataset
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# 1. Data Exploration
print("Dataset Overview:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())
print("\nData Types:")
print(df.info())

# 2. Feature Engineering
# Converting 'year' to 'car_age'
df['car_age'] = 2024 - df['year']

# Dropping irrelevant or redundant columns
df = df.drop(columns=['year', 'name'])

# 3. Model Selection and Training
# Separating features and target variable
X = df.drop(columns=['selling_price'])
y = df['selling_price']

# Identifying categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define the models
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the Random Forest model
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Model")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
cv_rmse_rf = np.mean(np.sqrt(-cross_val_score(rf_model, X, y, scoring='neg_mean_squared_error', cv=5)))
print("Cross-Validated RMSE:", cv_rmse_rf)
print("="*30)

# Train and evaluate the Gradient Boosting model
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("\nGradient Boosting Model")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_gb))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_gb)))
cv_rmse_gb = np.mean(np.sqrt(-cross_val_score(gb_model, X, y, scoring='neg_mean_squared_error', cv=5)))
print("Cross-Validated RMSE:", cv_rmse_gb)
print("="*30)

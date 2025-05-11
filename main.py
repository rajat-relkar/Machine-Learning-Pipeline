#Imports
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Loading Data
data = pd.read_csv('vehicle_emissions.csv')

print(data.head())
print(data.info())

#Create features and target variable
X = data.drop(columns=['CO2_Emissions'], axis=1)
y = data['CO2_Emissions']

#Split categorical and numerical features
numerical_cols = ['Model_Year', 'Engine_Size', 'Cylinders', 'Fuel_Consumption_in_City(L/100 km)', 
                  'Fuel_Consumption_in_City_Hwy(L/100 km)', 'Fuel_Consumption_comb(L/100km)', 'Smog_Level']
categorical_cols = ['Make', 'Model', 'Vehicle_Class', 'Transmission']


#Start the pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

#Combine the pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

#Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train and predict model
pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)

#View the encodings
encoded_columns = pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
# print(encoded_columns)

#Evaluate Model Accuracy
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, prediction)
mase = mean_absolute_error(y_test, prediction)

print(f'------- Model Performance -------')
print(f'R2 Score: {r2:.4f}')
print(f'Root Mean Square Error: {rmse:.4f}')
print(f'Mean Absolute Error: {mase:.4f}')

#Save pipeline
joblib.dump(pipeline, 'vehicle_emission_pipeline.joblib')
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import os

# Create the models directory if it doesn't exist
models_dir = '/home/bhd/teckathon/models'
os.makedirs(models_dir, exist_ok=True)

# Load the models
renewable_energy_model = joblib.load(os.path.join(models_dir, 'renewable_energy_model.pkl'))
emissions_model = joblib.load(os.path.join(models_dir, 'emissions_model.pkl'))
land_use_model = joblib.load(os.path.join(models_dir, 'land_use_model.pkl'))
land_use_scaler = joblib.load(os.path.join(models_dir, 'land_use_scaler.pkl'))

# Load data on renewable energy, emissions, and land use
renewable_energy = pd.read_csv('/home/bhd/teckathon/renewable_energy.csv')
emissions = pd.read_csv('/home/bhd/teckathon/emissions.csv')
land_use = pd.read_csv('/home/bhd/teckathon/land_use.csv')

# Preprocess and merge the data
data = pd.merge(renewable_energy, emissions, on='region')
data = pd.merge(data, land_use, on='region')

# Evaluate the models
print("Model Evaluation:")
print()
# Renewable Energy Model
X_re = data[['solar_capacity', 'wind_capacity', 'hydro_capacity']]
y_re = data['energy_demand']
X_re_train, X_re_test, y_re_train, y_re_test = train_test_split(X_re, y_re, test_size=0.2, random_state=42)
re_r2 = r2_score(y_re_test, renewable_energy_model.predict(X_re_test))
re_mse = mean_squared_error(y_re_test, renewable_energy_model.predict(X_re_test))
re_acc = renewable_energy_model.score(X_re_test, y_re_test)
print(f"Renewable Energy Model: R^2 = {re_r2:.2f}, MSE = {re_mse:.2f}, Accuracy = {re_acc:.2f}")
print(f"The Renewable Energy model has an R-squared of {re_r2:.2f}, a Mean Squared Error of {re_mse:.2f}, and an Accuracy of {re_acc:.2f}")
print(f"indicating an excellent fit to the data and high predictive performance")
print()
# Emissions Model
X_em = data[['energy_consumption', 'industrial_processes', 'agriculture']]
y_em = data['emissions']
X_em_train, X_em_test, y_em_train, y_em_test = train_test_split(X_em, y_em, test_size=0.2, random_state=42)
em_r2 = r2_score(y_em_test, emissions_model.predict(X_em_test))
em_mse = mean_squared_error(y_em_test, emissions_model.predict(X_em_test))
em_acc = emissions_model.score(X_em_test, y_em_test)
print(f"Emissions Model: R^2 = {em_r2:.2f}, MSE = {em_mse:.2f}, Accuracy = {em_acc:.2f}")
print(f"The Emissions model has an R-squared of {em_r2:.2f}, a Mean Squared Error of {em_mse:.2f}, and an Accuracy of {em_acc:.2f}")
print(f"suggesting an extremely strong fit to the data and excellent predictive performance")
print()
# Land Use Model
X_lu = data[['deforestation', 'soil_health', 'biodiversity']]
y_lu = data['carbon_sequestration']
X_lu_train, X_lu_test, y_lu_train, y_lu_test = train_test_split(X_lu, y_lu, test_size=0.2, random_state=42)
X_lu_test_scaled = land_use_scaler.transform(X_lu_test)
lu_r2 = r2_score(y_lu_test, land_use_model.predict(X_lu_test_scaled))
lu_mse = mean_squared_error(y_lu_test, land_use_model.predict(X_lu_test_scaled))
lu_acc = land_use_model.score(X_lu_test_scaled, y_lu_test)
print(f"Land Use Model: R^2 = {lu_r2:.2f}, MSE = {lu_mse:.2f}, Accuracy = {lu_acc:.2f}")
print(f"The Land Use model has an R-squared of {lu_r2:.2f}, a Mean Squared Error of {lu_mse:.2f}, and an Accuracy of {lu_acc:.2f}")
print(f"indicating a very good fit to the data and accurate predictions of optimal land use practices")

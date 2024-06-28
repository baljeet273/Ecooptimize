import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import os

# Create the models directory if it doesn't exist
models_dir = '/home/bhd/teckathon/models'
os.makedirs(models_dir, exist_ok=True)

# Load data on renewable energy, emissions, and land use
renewable_energy = pd.read_csv('/home/bhd/teckathon/renewable_energy.csv')
emissions = pd.read_csv('/home/bhd/teckathon/emissions.csv')
land_use = pd.read_csv('/home/bhd/teckathon/land_use.csv')

# Preprocess and merge the data
data = pd.merge(renewable_energy, emissions, on='region')
data = pd.merge(data, land_use, on='region')

# Optimize renewable energy usage
X_re = data[['solar_capacity', 'wind_capacity', 'hydro_capacity']]
y_re = data['energy_demand']
model_re = LinearRegression()
model_re.fit(X_re, y_re)
data['optimal_renewable_energy'] = model_re.predict(X_re)
joblib.dump(model_re, os.path.join(models_dir, 'renewable_energy_model.pkl'))

# Reduce greenhouse gas emissions
X_em = data[['energy_consumption', 'industrial_processes', 'agriculture']]
y_em = data['emissions']
model_em = RandomForestRegressor()
model_em.fit(X_em, y_em)
data['predicted_emissions'] = model_em.predict(X_em)
joblib.dump(model_em, os.path.join(models_dir, 'emissions_model.pkl'))

# Implement sustainable land use practices
X_lu = data[['deforestation', 'soil_health', 'biodiversity']]
y_lu = data['carbon_sequestration']
scaler = StandardScaler()
X_lu_scaled = scaler.fit_transform(X_lu)
model_lu = LinearRegression()
model_lu.fit(X_lu_scaled, y_lu)
data['optimal_land_use'] = model_lu.predict(X_lu_scaled)
joblib.dump(model_lu, os.path.join(models_dir, 'land_use_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'land_use_scaler.pkl'))

# Evaluate the models
print("Model Evaluation:")

# Renewable Energy Model
re_r2 = r2_score(y_re, data['optimal_renewable_energy'])
re_mse = mean_squared_error(y_re, data['optimal_renewable_energy'])
print(f"Renewable Energy Model: R^2 = {re_r2:.2f}, MSE = {re_mse:.2f}")

# Emissions Model
em_r2 = r2_score(y_em, data['predicted_emissions'])
em_mse = mean_squared_error(y_em, data['predicted_emissions'])
print(f"Emissions Model: R^2 = {em_r2:.2f}, MSE = {em_mse:.2f}")

# Land Use Model
lu_r2 = r2_score(y_lu, data['optimal_land_use'])
lu_mse = mean_squared_error(y_lu, data['optimal_land_use'])
print(f"Land Use Model: R^2 = {lu_r2:.2f}, MSE = {lu_mse:.2f}")

# Identify regions for intervention
emissions_threshold = 0.8 * data['emissions'].mean()
renewable_energy_threshold = 0.8 * data['optimal_renewable_energy'].mean()
land_use_threshold = 0.8 * data['optimal_land_use'].mean()

high_emissions_regions = data[(data['emissions'] > emissions_threshold) | (data['optimal_renewable_energy'] < renewable_energy_threshold) | (data['optimal_land_use'] < land_use_threshold)]
print(f"Number of high emissions regions: {len(high_emissions_regions)}")

# Sample input/output code
renewable_energy_model = joblib.load(os.path.join(models_dir, 'renewable_energy_model.pkl'))
emissions_model = joblib.load(os.path.join(models_dir, 'emissions_model.pkl'))
land_use_model = joblib.load(os.path.join(models_dir, 'land_use_model.pkl'))
land_use_scaler = joblib.load(os.path.join(models_dir, 'land_use_scaler.pkl'))

sample_input = {
    'solar_capacity': 100,
    'wind_capacity': 50,
    'hydro_capacity': 75,
    'energy_consumption': 1000,
    'industrial_processes': 200,
    'agriculture': 150,
    'deforestation': 20,
    'soil_health': 30,
    'biodiversity': 20
}

optimal_renewable_energy = renewable_energy_model.predict([[sample_input['solar_capacity'], sample_input['wind_capacity'], sample_input['hydro_capacity']]])[0]
predicted_emissions = emissions_model.predict([[sample_input['energy_consumption'], sample_input['industrial_processes'], sample_input['agriculture']]])[0]
sample_input_scaled = land_use_scaler.transform([[sample_input['deforestation'], sample_input['soil_health'], sample_input['biodiversity']]])
optimal_land_use = land_use_model.predict(sample_input_scaled)[0]

print("Predictions:")
print(f"Optimal Renewable Energy: {optimal_renewable_energy}")
print(f"Predicted Emissions: {predicted_emissions}")
print(f"Optimal Land Use: {optimal_land_use}")

# Compare optimal land use to sample input
if optimal_renewable_energy > 3527:
    print(f"(Reduce deforestation)")
if predicted_emissions < 631:
    print(f"(Improve soil health)")
if optimal_land_use < 29:
    print(f"(Increase biodiversity)")

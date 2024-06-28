from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__, template_folder='/home/bhd/teckathon', static_folder='/home/bhd/teckathon/static')

# Load the pre-trained models
renewable_energy_model = joblib.load('renewable_energy_model.pkl')
emissions_model = joblib.load('emissions_model.pkl')
land_use_model = joblib.load('land_use_model.pkl')
land_use_scaler = joblib.load('land_use_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    solar_capacity = float(request.form['solar_capacity'])
    wind_capacity = float(request.form['wind_capacity'])
    hydro_capacity = float(request.form['hydro_capacity'])
    energy_consumption = float(request.form['energy_consumption'])
    industrial_processes = float(request.form['industrial_processes'])
    agriculture = float(request.form['agriculture'])
    deforestation = float(request.form['deforestation'])
    soil_health = float(request.form['soil_health'])
    biodiversity = float(request.form['biodiversity'])

    optimal_renewable_energy = renewable_energy_model.predict([[solar_capacity, wind_capacity, hydro_capacity]])[0]
    predicted_emissions = emissions_model.predict([[energy_consumption, industrial_processes, agriculture]])[0]
    sample_input_scaled = land_use_scaler.transform([[deforestation, soil_health, biodiversity]])
    optimal_land_use = land_use_model.predict(sample_input_scaled)[0]

    recommendations = []
    if optimal_renewable_energy > 3527:
        recommendations.append("Reduce deforestation")
    if predicted_emissions < 631:
        recommendations.append("Improve soil health")
    if optimal_land_use < 29:
        recommendations.append("Increase biodiversity")

    return render_template('result.html', optimal_renewable_energy=optimal_renewable_energy, predicted_emissions=predicted_emissions, optimal_land_use=optimal_land_use, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
 

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the dataset into a DataFrame
df = pd.read_csv('power.csv')

# Calculate the average power consumption across all three zones
df['Average_PowerConsumption'] = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']].mean(axis=1)

# Define feature names
feature_names = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'Month', 'Day', 'Hour', 'Minute']

# Split the dataset into features (X) and target variable (y)
X = df[feature_names]
y = df['Average_PowerConsumption']

# Load the trained model from the pkl file
rf_model = joblib.load('rf_model.pkl')

# Make sure to provide feature names during model prediction
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        wind_speed = float(request.form['WindSpeed'])
        general_diffuse_flows = float(request.form['GeneralDiffuseFlows'])
        diffuse_flows = float(request.form['DiffuseFlows'])
        month = int(request.form['Month'])
        day = int(request.form['Day'])
        hour = int(request.form['Hour'])
        minute = int(request.form['Minute'])

        # Make prediction
        input_data = [[temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows, month, day, hour, minute]]
        prediction = rf_model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset into a DataFrame
df = pd.read_csv('power.csv')

# Calculate the average power consumption across all three zones
df['Average_PowerConsumption'] = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']].mean(axis=1)

# Split the dataset into features (X) and target variable (y)
X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'Month', 'Day', 'Hour', 'Minute']]
y = df['Average_PowerConsumption']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", score)

# Convert R-squared score to percentage accuracy
percentage_accuracy = score * 100
print("Percentage Accuracy:", percentage_accuracy)

# Dump the trained model to a pickle file
joblib.dump(rf_model, 'rf_model.pkl')

# Now, to make predictions for new data provided by the user:
# Assume `unseen_data` is a DataFrame containing new data provided by the user
# For demonstration purposes, let's create a sample unseen data
unseen_data = pd.DataFrame({
    'Temperature': [7.0],
    'Humidity': [70.0],
    'WindSpeed': [0.1],
    'GeneralDiffuseFlows': [0.06],
    'DiffuseFlows': [0.09],
    'Month': [5],
    'Day': [9],
    'Hour': [12],
    'Minute': [20]
})

# Use the trained model to make predictions for unseen data
predicted_power_consumption = rf_model.predict(unseen_data)

# Display the predicted power consumption
print("Predicted Average Power Consumption:", predicted_power_consumption)

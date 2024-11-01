import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
from supabase import create_client, Client

# Supabase credentials
url = "https://rdneyqtfvplosdpunipr.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJkbmV5cXRmdnBsb3NkcHVuaXByIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAxMzg1MDEsImV4cCI6MjA0NTcxNDUwMX0.pB02wlLEHrhU3BKMOq0RsV6J4ttUfa23GiQmlGE-KAQ"
supabase: Client = create_client(url, key)

# Load the dataset (use the correct path for Raspberry Pi)
df = pd.read_csv('/home/kudama/Downloads/Copy_2.csv')

# Remove outliers using IQR
Q1 = df['Battery Voltage'].quantile(0.25)
Q3 = df['Battery Voltage'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df_filtered = df[(df['Battery Voltage'] >= lower_bound) & (df['Battery Voltage'] <= upper_bound)]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_filtered['voltage_measured_scaled'] = scaler.fit_transform(df_filtered['Battery Voltage'].values.reshape(-1, 1))

# Prepare the time series data
def create_dataset(data, time_steps=1):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        Y.append(data[i + time_steps, 0])
    return np.array(X), np.array(Y)

time_steps = 60 # Number of time steps to look back
X, Y = create_dataset(df_filtered['voltage_measured_scaled'].values.reshape(-1, 1), time_steps)

# Reshape X for RNN input [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

# Load the pre-trained TensorFlow model
model_path = '/home/kudama/Downloads/lstm_model.h5'  # Ensure model is at this path
model = load_model(model_path)

# Perform inference on the test data
Y_pred = model.predict(X_test)

# Inverse transform the predictions and the true values to original scale
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_inv = scaler.inverse_transform(Y_pred)

# Calculate metrics
mse = mean_squared_error(Y_test_inv, Y_pred_inv)
mae = mean_absolute_error(Y_test_inv, Y_pred_inv)
rmse = math.sqrt(mse)
mape = np.mean(np.abs((Y_test_inv - Y_pred_inv) / Y_test_inv)) * 100

# Print the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Get the last predicted value of Battery Voltage
last_predicted_value = float(Y_pred_inv[-1][0])

# Check if it needs maintenance or is healthy
future_health = "Need Maintenance" if last_predicted_value < 36 else "Healthy"

# Print the status
print(f"Last predicted Battery Voltage: {last_predicted_value} V")
print(f"Future Health Status: {future_health}")

# Get the current date and time
current_time = datetime.now().strftime("%I:%M %p %m/%d/%y")

# Prepare data to insert into Supabase
prediction_data = {
    "future_voltage": last_predicted_value,
    "datetime": current_time,
    "future_health": future_health
}

# Store the data in Supabase
try:
    response = supabase.table("future_prediction").insert(prediction_data).execute()
    if response.status_code == 201:
        print("Data successfully written to Supabase.")
    else:
        print(f"Failed to write to Supabase: {response.error_message}")
except Exception as e:
    print(f"An error occurred: {e}")

# Plot the results
plt.plot(Y_test_inv, label='Actual')
plt.plot(Y_pred_inv, label='Predicted')
plt.legend()
plt.show()

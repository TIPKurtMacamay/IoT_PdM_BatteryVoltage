import board
import time
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import csv
import os
from time import sleep, strftime
import shutil
import subprocess  # Import subprocess to call the model_prediction script

# Number of readings to average
num_readings = 5

R1 = 30000.0
R2 = 14100.0
R1_1 = 10000.0
R2_2 = 2500.0

ref_voltage = 5.0

# Initialize the I2C interface
i2c = busio.I2C(board.SCL, board.SDA)

# Create an ADS1115 object
ads = ADS.ADS1115(i2c)

# Define the analog input channel
channel = AnalogIn(ads, ADS.P0)

# File paths
file_path = 'sensor_data.csv'
history_file_path = 'history_data.csv'

# Check if the CSV file exists and has a header
header_exists = os.path.isfile(file_path)
if header_exists:
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header_exists = next(reader, None) == ['Timestamp', 'Battery Voltage']

# Open the CSV file in append mode
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row if it doesn't exist
    if not header_exists:
        writer.writerow(['Timestamp',  'Battery Voltage'])
        file.flush()  # Ensure the header is written immediately

    # Initialize `previous_voltage` to 0
previous_voltage = 0.0  # Start with a default numeric value
voltage_increment_count = 0  # Counter for voltage increments

timer_start_time = None  # Start time for the 30 mins timer
last_change_time = None  # Last time the voltage changed

# Loop to read the analog input continuously
while True:
    try:
        # Get ADC value and calculate voltages
        adc_value = channel.value
        adc_voltage = adc_value * 1.875 / 1000.0
        in_voltage = adc_voltage / (R2 / (R1 + R2))
        orig_voltage = in_voltage / (R2_2 / (R1_1 + R2_2))
        
        # Print the readings
        print("Battery Voltage: ", orig_voltage)
        print("\n")

        # Check if the voltage has changed by 0.1V or more
        if abs(orig_voltage - previous_voltage) >= 0.1:
            # Update the last change time
            last_change_time = time.time()

            # Start or continue the 30 minutes timer
            if timer_start_time is None:
                timer_start_time = time.time()
                print("30-minute timer started.")

            # Check if the voltage has decreased by 0.1V or more
            if orig_voltage < previous_voltage - 0.1:
                # Get the current timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Write the data to the CSV file
                writer.writerow([timestamp, orig_voltage])
                file.flush()  # Ensure the data is written to disk
                print("Data written to CSV file.")
            
            # Check if the voltage has increased by 0.1V or more
            if orig_voltage >= previous_voltage + 0.1:
                voltage_increment_count += 1
                print(f"Voltage increased by 0.1V or more {voltage_increment_count} times.")

                if voltage_increment_count >= 25:
                    print("Voltage increased by 0.1V or more 25 times. Copying data to history_data.csv.")
                    with open(file_path, mode='r') as src, open(history_file_path, mode='a', newline='') as dst:
                        reader = csv.reader(src)
                        writer_dst = csv.writer(dst)
                        # Write data to history_data.csv
                        for row in reader:
                            writer_dst.writerow(row)

                    # Clear the original file except the header
                    with open(file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Timestamp', 'Battery Voltage'])
                    print("Data cleared from sensor_data.csv except the header.")
                    
                    # Reset the counter
                    voltage_increment_count = 0

        # If the voltage hasn't changed for 5 seconds, stop the timer
        if last_change_time and time.time() - last_change_time >= 5:
            print("Voltage didn't change for 5 seconds. Stopping the 30-minute timer.")
            timer_start_time = None
            last_change_time = None

        # If the timer has reached 30 minutes, perform the desired action
        if timer_start_time and time.time() - timer_start_time >= 30 * 60:
            print("30 minutes have passed with changing voltage. Running model_prediction.py.")
            # Run model_prediction.py after 30 minutes have passed
            subprocess.run(["python3", "model_prediction.py"])
            
            # Reset the timer
            timer_start_time = None

        # Update previous voltage
        previous_voltage = orig_voltage
        
        # Sleep for 1 second
        time.sleep(1)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        break

import pandas as pd
import matplotlib.pyplot as plt

# Replace 'data.csv' with your CSV file path
csv_file = 'sensor_data.csv'

# Read the CSV file
data = pd.read_csv(csv_file)

# Replace with the actual column names you want to plot
column_name1 = 'Battery Voltage'
column_name2 = 'Temperature (C)'

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Check if the first column exists and plot it
if column_name1 in data.columns:
    column_data1 = data[column_name1]
    ax1.plot(column_data1, marker='o', linestyle='-')
    ax1.set_title(f'Plot of {column_name1}')
    ax1.set_xlabel('Index')
    ax1.set_ylabel(column_name1)
    ax1.grid(True)
else:
    print(f'Column "{column_name1}" does not exist in the CSV file.')

# Check if the second column exists and plot it
if column_name2 in data.columns:
    column_data2 = data[column_name2]
    ax2.plot(column_data2, marker='x', linestyle='-')
    ax2.set_title(f'Plot of {column_name2}')
    ax2.set_xlabel('Index')
    ax2.set_ylabel(column_name2)
    ax2.grid(True)
else:
    print(f'Column "{column_name2}" does not exist in the CSV file.')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Save the plot to a file (optional)
# plt.savefig('plot.png')

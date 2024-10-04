import matplotlib.pyplot as plt
import numpy as np
import struct
import os
import pandas as pd
from datetime import datetime, timezone
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import tkinter as tk
from tkinter import filedialog, messagebox

# Main function to implement the algorithm sequence
def main(magnetometer_file, imu_file, save_folder, save_filename):
    # Parse magnetometer data into a DataFrame
    magnetometer_data = parse_magnetometer_data(magnetometer_file, pos_time=1, pos_field=4, rounding=3)
    filtered_data = magnetometer_data

    # Filter parameters for high-pass filtering to get the high-frequency component
    cutoff_freq = 0.01  # Cutoff frequency (Hz)
    sampling_freq = 1  # Sampling frequency (Hz)
    filter_order = 4  # Filter order

    # Apply high-pass filter and create a new column with the filtered high-frequency component
    filtered_data['Filtered_Field'] = high_pass_filter(filtered_data['Field'], cutoff_freq, sampling_freq, filter_order)
    
    # Get the date from the first row to correctly calculate the IMU data time
    first_date = filtered_data['UTC_date'].iloc[0]
    date_obj = datetime.strptime(first_date, '%d-%m-%Y').replace(tzinfo=timezone.utc)

    # Parse IMU file
    imu_data = parse_imu_data(date_obj, imu_file)
    
    # Associate each magnetometer measurement with a correction based on IMU data
    combined_data = concatenate_data(filtered_data, imu_data)
    
    # Determine the normalization coefficient based on correction and high-frequency component of the field
    mid_data = combined_data[len(combined_data)//4 : -len(combined_data)//4]  # Use central half of the data
    correction = np.array(mid_data['Correction'])
    filtered_field = np.array(mid_data['Filtered_Field'])

    # Use MSE minimization to get the best multiplier for the correction
    result = minimize(calculate_mse, 0.01, args=(correction, filtered_field))
    optimal_coefficient = result.x[0]

    # Calculate corrected field
    combined_data['Corrected_Field'] = combined_data['Correction'] * optimal_coefficient + combined_data['Field']
    
    # Drop unnecessary columns and prepare data for output
    combined_data = combined_data.drop(columns=['Filtered_Field', 'Time', 'Correction'])
    
    # Save DataFrame to a text file with tab as delimiter
    combined_data.to_csv(os.path.join(save_folder, f"{save_filename}.txt"), sep='\t', index=False) 

# Function to select a file and insert its path into the text entry field
def select_file(entry):
    file_path = filedialog.askopenfilename()
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

# Function to select a folder and insert its path into the text entry field
def select_folder(entry):
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry.delete(0, tk.END)
        entry.insert(0, folder_path)

# Function to run the main function with the user input values
def run_function():
    magnetometer_file = entry1.get()
    imu_file = entry2.get()
    save_folder = entry_folder.get()
    save_filename = entry_filename.get()

    if not magnetometer_file or not imu_file:
        messagebox.showwarning("Error", "Please select both files.")
    elif not save_folder or not save_filename:  
        messagebox.showwarning("Error", "Please select a folder and enter a filename.")
    else:
        main(magnetometer_file, imu_file, save_folder, save_filename)
        root.destroy()  # Close the window

# Function to minimize MSE
def calculate_mse(coefficient, correction, filtered_field):
    modified_field = filtered_field + coefficient * correction
    mean_modified_field = np.mean(modified_field)
    return np.mean((modified_field - mean_modified_field) ** 2)

# High-pass filter function
def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Remove NaN values before filtering
    if np.any(np.isnan(data)):
        data = np.interp(np.arange(len(data)), np.arange(len(data))[~np.isnan(data)], data[~np.isnan(data)])

    try:
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    except Exception as e:
        print(f"Error applying filter: {e}")
        return np.full_like(data, np.nan)

# Convert time to seconds from the beginning of the day
def time_to_seconds(utc_time):
    h, m, s = map(float, utc_time.split(':'))
    return h * 3600 + m * 60 + s

# Function to parse magnetometer data file
def parse_magnetometer_data(filename, pos_time=0, pos_field=1, rounding=3):
    df = pd.read_csv(filename, sep='\t')

    # Add a column with time from the beginning of the day
    df['Time'] = df['UTC_time'].apply(time_to_seconds)
    
    return df

# Function to parse IMU data file
def parse_imu_data(ref_datetime, filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            file_size = os.path.getsize(filename)
            
            block_size = 4 + 6 * 2  # 4 bytes for int32 and 6 * 2 bytes for int16
            num_blocks = file_size // block_size
            
            timer_data = np.zeros(num_blocks, dtype=np.uint32)
            imu_data = np.zeros((num_blocks, 6), dtype=np.int16)
            
            for i in range(num_blocks):
                timer_data[i] = struct.unpack('I', file.read(4))[0]
                imu_data[i, :] = struct.unpack('6h', file.read(12))
        
        ref_time_millis = ref_datetime.timestamp() * 1000
        time_high = int(ref_time_millis // 2**32) * 2**32
        imu_timestamps = time_high + timer_data
        
        imu_time_seconds = imu_timestamps / 1000.0 
        imu_relative_time = imu_time_seconds - ref_time_millis / 1000
        
        imu_table = pd.DataFrame({
            'Time': imu_relative_time,
            'gX': imu_data[:, 0],
            'gY': imu_data[:, 1],
            'gZ': imu_data[:, 2],
            'bX': imu_data[:, 3],
            'bY': imu_data[:, 4],
            'bZ': imu_data[:, 5],
        })
        return imu_table
    else:
        raise FileNotFoundError(f"File {filename} does not exist.")

# Function to concatenate magnetometer and IMU data
def concatenate_data(magnetometer_df, imu_df):
    c = 0
    num_rows = len(magnetometer_df)  
    correction = np.zeros(num_rows)

    for i in range(num_rows):
        mag_time = magnetometer_df.iloc[i]['Time']
        correction_measurements = []

        while c < len(imu_df) and imu_df.iloc[c]['Time'] < mag_time + 0.1:
            if imu_df.iloc[c]['Time'] >= mag_time + 0.05:
                row = imu_df.iloc[c]
                bX, bY, bZ, gX, gY, gZ = row['bX'], row['bY'], row['bZ'], row['gX'], row['gY'], row['gZ']
                
                # Get rotation matrix based on magnetic vector
                rotation_matrix = calculate_rotation_matrix(bX, bY, bZ)
                
                # Apply rotation matrix to gyroscope vector and get the new x component
                correction_measurements.append(apply_rotation_matrix(rotation_matrix, gX, gY, gZ) * 0.011)
            c += 1
        correction[i] = np.mean(correction_measurements)
    
    magnetometer_df['Correction'] = correction   
    return magnetometer_df

# Normalize vector
def normalize(vector):
    return vector / np.linalg.norm(vector)

# Calculate rotation matrix based on magnetic vector
def calculate_rotation_matrix(bx, by, bz):
    magnetic_vector = np.array([bx, by, bz])
    magnetic_vector_normalized = normalize(magnetic_vector)
    
    arbitrary_vector = np.array([0, 0, 1])
    if np.allclose(np.cross(magnetic_vector_normalized, arbitrary_vector), 0):
        arbitrary_vector = np.array([0, 1, 0])
    
    y1 = normalize(arbitrary_vector - np.dot(arbitrary_vector, magnetic_vector_normalized) * magnetic_vector_normalized)
    z1 = np.cross(magnetic_vector_normalized, y1)
    
    rotation_matrix = np.vstack([magnetic_vector_normalized, y1, z1])
    return rotation_matrix

# Apply rotation matrix to gyroscope vector
def apply_rotation_matrix(rotation_matrix, gx, gy, gz):
    gyroscope_vector = np.array([gx, gy, gz])
    rotated_vector = rotation_matrix @ gyroscope_vector
    return rotated_vector[0]

# Create the main window and add elements
root = tk.Tk()
root.title("Data Processor")

# Labels
tk.Label(root, text="Magnetometer file").grid(row=0, column=0)
tk.Label(root, text="IMU file").grid(row=1, column=0)
tk.Label(root, text="Save to folder").grid(row=2, column=0)
tk.Label(root, text="Save filename").grid(row=3, column=0)

# Entry fields
entry1 = tk.Entry(root, width=50)
entry1.grid(row=0, column=1)
entry2 = tk.Entry(root, width=50)
entry2.grid(row=1, column=1)
entry_folder = tk.Entry(root, width=50)
entry_folder.grid(row=2, column=1)
entry_filename = tk.Entry(root, width=50)
entry_filename.grid(row=3, column=1)

# Buttons
tk.Button(root, text="Select", command=lambda: select_file(entry1)).grid(row=0, column=2)
tk.Button(root, text="Select", command=lambda: select_file(entry2)).grid(row=1, column=2)
tk.Button(root, text="Select", command=lambda: select_folder(entry_folder)).grid(row=2, column=2)
tk.Button(root, text="Run", command=run_function).grid(row=4, column=1)

# Start main loop
root.mainloop()

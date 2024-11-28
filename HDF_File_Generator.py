import h5py
import numpy as np
import pandas as pd
import os
import re
import csv

# Path to the HDF5 file
DIR = r"D:\master thesis\Smart meter data\smd-new"

# To get all the files in the folder and filter them based on years
def classify_files_by_year(directory, year):
    files = []
    try:
        pattern = re.compile(str(year))
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)) and pattern.search(filename):
                files.append(filename)
        return files
    except FileNotFoundError:
        print(f"Directory {directory} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# To read the file and drop rows/columns
def read(file_path):
    rows_to_skip = ['** aggregate output **', 'Kundensegment', 'Tariftyp', 'Messpunkt', 'Profil-Nummer', 'GP-Nummer',
                    'Messpunkt', 'Profilbezeichnung']
    df = pd.read_csv(file_path, sep=';', header=None, low_memory=False)
    escaped_patterns = [re.escape(pattern) for pattern in rows_to_skip]
    pattern = '|'.join(escaped_patterns)

    df = df[~df.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)]

    df = df.drop(
        columns=[1])  # Drop the first column (index 1 since 0 is the actual first column if we keep header=None)
    df = df.dropna()  # Drop rows with any NaN values
    df.reset_index(drop=True, inplace=True)
    return df

# To merge all the files next to each other
def process_files(directory, year):
    files = classify_files_by_year(directory, year)
    if files:
        merged_df = None
        for idx, file in enumerate(files):
            file_path = os.path.join(directory, file)
            df = read(file_path)
            print(f"Processing file: {file}")
            print(df)  # Print the DataFrame content
            if idx == 0:
                merged_df = df
            else:
                df = df.iloc[:, 1:].reset_index(drop=True)
                merged_df = pd.concat([merged_df, df], axis=1)
        return merged_df
    return None

# To check each Anlage number and rolle number in merged df and create a separate dataframe for each anlage number
def convert(df):
    # Converting `Anlage` and `Rolle` to integers
    df.iloc[0, 1:] = df.iloc[0, 1:].astype(int)
    df.iloc[1, 1:] = df.iloc[1, 1:].astype(int)

    anlage_values = df.iloc[0, 1:].values
    rolle_values = df.iloc[1, 1:].values
    timestamps = df.iloc[2:, 0].values

    columns = ['Timestamp', 'A+', 'Ri+', 'A-', 'Ri-', 'Meter_id']
    result_df = pd.DataFrame(columns=columns)
    # Process each unique `Anlage`
    unique_anlage_numbers = pd.unique(anlage_values)

    for anlage in unique_anlage_numbers:
        # Initialize columns for A+, Ri+, A-, Ri- values
        a_plus = [0] * len(timestamps)
        r_plus = [0] * len(timestamps)
        a_minus = [0] * len(timestamps)
        r_minus = [0] * len(timestamps)

        # Get all the columns corresponding to the current `Anlage`
        relevant_columns = [i for i, val in enumerate(anlage_values) if val == anlage]
        # Assign values based on `Rolle` numbers for the current `Anlage`
        for i in relevant_columns:
            rolle = rolle_values[i]
            if rolle in [1, 71]:
                a_plus = df.iloc[2:, i + 1].values  # +1 to account for the Timestamp column
            elif rolle in [2, 72]:
                r_plus = df.iloc[2:, i + 1].values
            elif rolle in [3, 73]:
                a_minus = df.iloc[2:, i + 1].values
            elif rolle in [4, 74]:
                r_minus = df.iloc[2:, i + 1].values

        temp_df = pd.DataFrame({
            'Timestamp': timestamps,
            'A+': a_plus,
            'Ri+': r_plus,
            'A-': a_minus,
            'Ri-': r_minus,
            'Meter_id': [anlage] * len(timestamps)
        })
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
    result_df.reset_index(drop=True, inplace=True)
    return result_df,unique_anlage_numbers


def filter_and_store_hdf5(df, meter_id,year):
    with pd.HDFStore(f'hdf_file_{year}.h5', mode='w') as store:
        meter_id.sort()
        for value in meter_id:
            filtered_df = df[df['Meter_id'] == value]
            store.put(str(value), filtered_df)
            # Saving individual csv files ( optional, it will create individual csv file for each analge id)
            #filtered_df.to_csv(f"D:\\master thesis\\Smart meter data\\smartmeter-main\\smartmeter-main\\smartmeter\\{year}\\{value}.csv")

# To read the hdf_file
def read_hdf_file(hdf5_file_path,file_id):
    try:
        with pd.HDFStore(hdf5_file_path, 'r') as hdf_store:
            if file_id is not None:
                df = hdf_store[f'/{file_id}']
                df['A+'] = pd.to_numeric(df['A+'], errors='coerce')
                df['A-'] = pd.to_numeric(df['A-'], errors='coerce')
                df.reset_index()
                return df
            else:
                # Get all keys (analge IDs) in the HDF5 file
                available_keys = list(hdf_store.keys())
                for key in available_keys:
                    df = hdf_store[key]
                    required_columns = ['Timestamp', 'A+', 'Ri+', 'A-', 'Ri-', 'Meter_id']
                    for col in required_columns:
                        if col not in df.columns:
                            raise ValueError(f"Column '{col}' is missing in the DataFrame for meter ID '{key}'")
                    df['A+'] = pd.to_numeric(df['A+'], errors='coerce')
                    df['A-'] = pd.to_numeric(df['A-'], errors='coerce')
                    return  df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":

    year = 2022
    merged_df = process_files(DIR,year) # merge all the files for same year together in one DF (done in case one anlage number in multiple files)
    hdf_df, meter_id= convert(merged_df) # check all columns belonging to one Anlage number
    filter_and_store_hdf5(hdf_df, meter_id, year) # create one hdf file

    year = 2023
    merged_df = process_files(DIR,year) # merge all the files for same year together in one DF (done in case one anlage number in multiple files)
    hdf_df, meter_id= convert(merged_df) # check all columns belonging to one Anlage number
    filter_and_store_hdf5(hdf_df, meter_id, year) # create one hdf file for the year

    year = 2024
    merged_df = process_files(DIR, year)  # merge all the files for same year together in one DF (done in case one anlage number in multiple files)
    hdf_df, meter_id = convert(merged_df)  # check all columns belonging to one Anlage number
    filter_and_store_hdf5(hdf_df, meter_id, year)  # create one hdf file for the year

    # Function to read hdf_file
    hdf5_file_path = 'hdf_file_2023.h5'
    # If file_id is given then it will generate that file else if file_id none then extract all
    result_df = read_hdf_file(hdf5_file_path, file_id = '712834')
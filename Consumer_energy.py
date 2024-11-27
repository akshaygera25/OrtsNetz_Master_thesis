import h5py
import pandas as pd
import csv
import os
from dataloaders import AIMLoader
import numpy as np
import plot_utils as plot_utils
from scipy.stats import mannwhitneyu

df_map = pd.read_csv('D:\\master thesis\\Smart meter data\\mapping.csv', delimiter=';')
dir = "D:\\master thesis\\Smart meter data"
dyn_dir = 'D:\\master thesis\\Data\\price-dynamic.csv'
enrollment_dir= 'D:\\master thesis\\Smart meter data\\all_customer_data_2024-06-13_reduced.xlsx'

# For daily averaged plots, must contain either 2022,2023,2024 data or 2023,2024 data
load_data = 'D:\\master thesis\\Smart meter data\\smartmeter-main\\smartmeter-main\\smartmeter\\Load Data\\Winter_Dec-Feb'


# To identify the ldn id with the mentioned operend_codes
def operncode():
    df_map = pd.read_csv(os.path.join(dir, 'mapping.csv'), sep=';')
    filtered_df = df_map[
        df_map['SBEH-DIR'].notna() & (df_map['SBEH-DIR'] != 0) & (df_map['SBEH-DIR'] != '') |
        df_map['SBEH-SP'].notna() & (df_map['SBEH-SP'] != 0) & (df_map['SBEH-SP'] != '')
        ]
    flagged= filtered_df['aim_geraet_ldn'].to_list()
    return flagged

# Function use to get ldn ids as list from B.P. numbers (dataframe)
# Used because original B.P.No dataframe is created using Enrollment data sheet which has B.P.No only
def map_filter(df):
    bp_no = df['Geschäftspartner'].unique()
    filtered_df_map = df_map[df_map['Geschäftspartner'].isin(bp_no)]
    aim_geraet_ldn_list = filtered_df_map['aim_geraet_ldn'].dropna().tolist()
    aim_geraet_ldn_list = [int(i) for i in aim_geraet_ldn_list]
    return aim_geraet_ldn_list

# Function to convert slot number(1,2,3,...) to time(00:00, 00:15, 00:30....)
def slot_to_time(mean_prof_build):
    # Use zero-based indexing: slot - 1
    slot_zero_indexed = mean_prof_build - 1
    hours = slot_zero_indexed // 4
    minutes = (slot_zero_indexed % 4) * 15
    return f"{hours:02}:{minutes:02}"


#    This function loads the HDF5 file and AIMLoader and returns the loader and list of ldn numbers.
def read():
    with h5py.File(os.path.join(dir, 'database.hdf5'), 'r') as file:
        # Getting all the ldn numbers from HDF5 file
        hdf_old = [int(x) for x in file.keys()]

    with h5py.File(os.path.join(dir, 'database-with-sql.hdf5'), 'r') as file:
        # Getting all the ldn numbers from HDF5 file
        hdf_new = [int(x) for x in file.keys()]
    # Initialize the AIMLoader
    dl_all = AIMLoader(
        os.path.join(dir, 'database.hdf5'),
        os.path.join(dir, 'mapping.csv')
    )
    dl_new = AIMLoader(
        os.path.join(dir, 'database-with-sql.hdf5'),
        os.path.join(dir, 'mapping.csv')
    )
    hdf = list(dict.fromkeys(hdf_old + hdf_new))
    return dl_all,dl_new, hdf_old,hdf_new,hdf   # hdf contains both new and old ids


#   This function iterates through the hdf list and calculates the energy consumption for each file id.
def energy(dl_all, dl_new, hdf_new, hdf, start_date, end_date):
    ldn = {}

    for i, file_id in enumerate(hdf):
        # If id is found in new file then use this otherwise jump to next file
        if file_id in hdf_new:
            try:
                ldn[file_id] = yearlysum(dl_new, file_id, start_date, end_date)
                print(file_id, i, "passed first")  # For troubleshooting
            except Exception as e:
                print(f"Error occurred for {file_id}: {e}. Skipping to the next iteration.")
                continue
        else:
            try:
                ldn[file_id] = yearlysum(dl_all, file_id, start_date, end_date)
                print(file_id, i, "passed second")  # For troubleshooting
            except Exception as e:
                print(f"Error occurred for {file_id}: {e}. Skipping to the next iteration.")
                continue
    return ldn


# Used for calculating the energy during high tariff and low tariffs as per regular/old tariffs scheme
def yearlysum(dl, file_id,start_date,end_date):
    # loading data
    df = dl.load(file_id)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    # Extracting date components
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['Hour'] = df['Timestamp'].dt.hour

    # Calculating active and reactive power
    df['ActiveP'] = df['A+'] - df['A-']
    df['ReactiveP'] = df['Rc-'] - df['Ri+']

    # Removing timezone information
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

    # Creating the 'Year_Month' column
    df['Year_Month'] = df['Timestamp'].dt.to_period('M')

    # Determine the year for filtering
    year = pd.to_datetime(start_date).year
    if year == 2022:
        conditions = ((df['Weekday'].between(0, 4)) & (df['Hour'].between(7, 19))) | \
                     ((df['Weekday'] == 5) & (df['Hour'].between(7, 12)))
    elif year in [2023, 2024]:
        conditions = (df['Weekday'].between(0, 4)) & (df['Hour'].between(7, 19))
    else:
        conditions = pd.Series([False] * len(df))

    # Assigning the 'Tariff' column based on conditions
    df['Tariff'] = np.where(conditions, 'High', 'Low')

    # Grouping by 'Year_Month' and calculating high and low energy
    high_energy = df[df['Tariff'] == 'High'].groupby('Year_Month')['ActiveP'].sum()
    low_energy = df[df['Tariff'] == 'Low'].groupby('Year_Month')['ActiveP'].sum()

    # Combining high and low energy into a single DataFrame
    monthly_energy = pd.concat([high_energy, low_energy], axis=1).reset_index()
    monthly_energy.columns = ['Year_Month', 'high_energy', 'low_energy']

    # Converting 'Year_Month' to timestamp for filtering
    monthly_energy['Year_Month'] = monthly_energy['Year_Month'].dt.to_timestamp()

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filtering data based on the provided start_date and end_date
    target_energy = monthly_energy.loc[(monthly_energy['Year_Month'] >= start_date) &
                                       (monthly_energy['Year_Month'] <= end_date)]

    # Calculate sums of high and low energy
    sum_high_energy = target_energy['high_energy'].sum()
    sum_low_energy = target_energy['low_energy'].sum()
    return [sum_high_energy, sum_low_energy]


# Function to merge high/low energy returned from yearlysum dict into dataframe with ldn ids and mapping details
# Same function also used for merging if data is with OrtsNetz_TOU (with medium energy column)
def merge_highlow_mapp(dfs,ldn_dict,high_low= True ):
    df2 = pd.read_csv(os.path.join(dir, 'mapping.csv'), sep=';')
    df2 = df2[['Anlagenart', 'aim_geraet_ldn', 'Geschäftspartner', 'nis_name','Anlage']]
    df2 = df2.dropna(subset=['aim_geraet_ldn'])
    all_ldn_data = []

    key = next(iter(ldn_dict))
    value = ldn_dict[key]
    if isinstance(value, list):
        if len(value) == 2:
            medium = False
        elif len(value) == 3:
            medium = True

    if high_low:
        df2['high_energy'] = None
        df2['low_energy'] = None
        if medium == True:
            df2['medium_energy'] = None
        for index, row in df2.iterrows():
            key = row['aim_geraet_ldn']
            if key in ldn_dict:
                if medium == True:
                    high_energy,medium_energy, low_energy = ldn_dict[key]
                else:
                    high_energy, low_energy = ldn_dict[key]

                df2.at[index, 'high_energy'] = high_energy
                if medium == True:
                    df2.at[index, 'medium_energy'] = medium_energy
        print(df2)
        df2.at[index, 'low_energy'] = low_energy
        # Calculating total energy meter_wise for normalisation
        df2['Total_Energy'] = df2['low_energy'] + df2['high_energy']
    else:
        for ldn, df in ldn_dict.items():
            df['aim_geraet_ldn'] = ldn
            all_ldn_data.append(df)

        combined_ldn_data_df = pd.concat(all_ldn_data, ignore_index=True)
        # Merge the DataFrames on 'aim_geraet_ldn'
        merged_df = pd.merge(combined_ldn_data_df, df2, on='aim_geraet_ldn', how='inner')
        merged_df = mapping(merged_df, dfs)
        df2 = merged_df
        # To get building type mean profile
        grouped_data = merged_df.groupby(['Anlagenart', 'slot'])['ActiveP'].mean().reset_index()
        # To get avg category profile
        grouped_data_cat = merged_df.groupby('slot')['ActiveP'].mean().reset_index()
    if high_low == True:
        df2['aim_geraet_ldn'] = df2['aim_geraet_ldn'].astype(str)
        df2['nis_name'] = df2['nis_name'].astype(str)
        df2['Anlagenart'] = df2['Anlagenart'].astype(str)
        df2['Anlage'] = df2['Anlage'].astype(str)
        agg_dict = {
            'aim_geraet_ldn': ','.join,
            'Anlage': ','.join,
            'nis_name': ','.join,
            'Anlagenart': ','.join,
            'high_energy': 'sum',
            'low_energy': 'sum'
        }
        # Modify the aggregation dictionary if 'medium_energy' exists in the DataFrame
        if 'medium_energy' in df2.columns:
            agg_dict['medium_energy'] = 'sum'
        # Group by 'Geschäftspartner' and apply the aggregation
        grouped_df = df2.groupby('Geschäftspartner').agg(agg_dict).reset_index()
        return grouped_df
    else:
        return grouped_data, grouped_data_cat, merged_df


# To map based on category i.e. Category as Opt_in_TOU/DYN, Not_opt_out_TOU/DYN and control as Auto, Manual
def mapping(df,dfs):
    df_optin_dyn, df_optin_tou, df_notoptout_dyn, df_notoptout_tou, df_auto_wout_dir, df_manu = dfs
    optin_dyn_set = set(df_optin_dyn['Geschäftspartner'])
    optin_tou_set = set(df_optin_tou['Geschäftspartner'])
    notoptout_dyn_set = set(df_notoptout_dyn['Geschäftspartner'])
    notoptout_tou_set = set(df_notoptout_tou['Geschäftspartner'])
    auto_wout_dir_set = set(df_auto_wout_dir['Geschäftspartner'])
    manu_set = set(df_manu['Geschäftspartner'])
    def determine_keyword(x):
        if x in optin_dyn_set:
            return 'Optin Dyn'
        elif x in optin_tou_set:
            return 'Optin Tou'
        elif x in notoptout_dyn_set:
            return 'Not opt out Dyn'
        elif x in notoptout_tou_set:
            return 'Not opt out Tou'
        else:
            return ''
    def determine_type(x):
        if x in auto_wout_dir_set:
            return 'Auto'
        elif x in manu_set:
            return 'Manual'
        else:
            return ''
    df['Category'] = df['Geschäftspartner'].apply(determine_keyword)
    df['Control'] = df['Geschäftspartner'].apply(determine_type)
    return df



# Tariffs as per regular tariffs
def tariffs_regular(df_eng,year):
    """
    This function calculates energy costs based on predefined tariffs.
    """
    default_values = {
        2024: (19, 17.50, 6.90, 6.90, 4.41,8),
        2023: (11.45, 10.35, 7.6, 4.7,2.9205 ,8),
        2022: (7.25, 6.15, 7.30, 4.10,2.62 ,8),
        2021: (7.55, 6.20, 7.45, 3.7, 2.62 ,8)
    }

    if year not in default_values:
        raise ValueError("Prices for the given year are not available")

    df_eng['aim_geraet_ldn'] = df_eng['aim_geraet_ldn'].astype(str)
    df_eng['nis_name'] = df_eng['nis_name'].astype(str)
    df_eng['Anlagenart'] = df_eng['Anlagenart'].astype(str)

    Energy_high, Energy_low, Network_high, Network_low, other_cost, baseprice = default_values[year]

    grouped_df = df_eng.groupby('Geschäftspartner').agg({
        'aim_geraet_ldn': ','.join,
        'nis_name': ','.join,
        'Anlagenart': ','.join,
        'high_energy': 'sum',
        'low_energy': 'sum'
    }).reset_index()
    grouped_df['ActiveP'] = grouped_df['high_energy'] + grouped_df['low_energy']
    grouped_df['Energy_high_cost'] = grouped_df['high_energy'] * Energy_high * 0.01
    grouped_df['Energy_low_cost'] = grouped_df['low_energy'] * Energy_low * 0.01
    grouped_df['Network_high_cost'] = grouped_df['high_energy'] * Network_high * 0.01
    grouped_df['Network_low_cost'] = grouped_df['low_energy'] * Network_low * 0.01
    grouped_df['Network_cost'] = grouped_df['Network_high_cost'] + grouped_df['Network_low_cost']
    grouped_df['Energy_cost'] = grouped_df['Energy_low_cost'] + grouped_df['Energy_high_cost']
    grouped_df['Other_cost'] = grouped_df['ActiveP'] * other_cost * 0.01
    def calculate_total_price(row):
        num_list = row['aim_geraet_ldn'].split(',')
        # Count the number of elements in the list
        num_count = len(num_list)
        # Calculate the total price
        return num_count * baseprice

    # Apply the function to each row in the DataFrame
    grouped_df['Basic_cost'] = grouped_df.apply(calculate_total_price, axis=1)
    grouped_df['Total_cost'] = grouped_df['Energy_cost'] + grouped_df['Network_cost'] + grouped_df['Other_cost'] + grouped_df['Basic_cost']
    grouped_df = grouped_df[grouped_df['Geschäftspartner'] != 0]
    grouped_df.to_excel(f'energy&costs_{year}.xlsx')
    return grouped_df

# Function to calculated daily quarter hourly averaged energy consumption
def dailyprofile_data(dl, file_id,start_date=None,end_date=None):
    df = dl.load(file_id)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df['date'] = df['Timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['Timestamp'].dt.strftime('%H:%M')
    df['slot'] = df['Timestamp'].dt.hour * 4 + df['Timestamp'].dt.minute // 15 + 1
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['Hour'] = df['Timestamp'].dt.hour
    df['ActiveP'] = (df['A+'] - df['A-'])*4
    df['ReactiveP'] = (df['Rc-'] - df['Ri+'])*4
    filter_df = df[['date','time','Hour','Weekday','slot','ActiveP','ReactiveP']]
    if start_date and end_date != None:
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        filter_df = filter_df.loc[df['date'].apply(lambda d: start_date <= d.date()<= end_date)]
    elif start_date!= None:
        start_date = pd.to_datetime(start_date).date()
        filter_df = filter_df.loc[df['date'].apply(lambda d: start_date <= d.date())]
    elif end_date != None:
        end_date = pd.to_datetime(end_date).date()
        filter_df = filter_df.loc[df['date'].apply(lambda d: d.date() <= end_date)]
    average_activep_profile = filter_df.groupby('slot')['ActiveP'].mean().reset_index()
    return average_activep_profile


# Function to call for daily profiles
# This function iterates through the hdf list and calculates the energy consumption for each file id.
def dailyprofile(dfs,dl_all,dl_new,lst,hdf_new,title, start_date=None,end_date=None,year=2022):

    ldn = {}
    for i, file_id in enumerate(lst):
        if file_id in hdf_new:
            try:
                ldn[file_id] = dailyprofile_data(dl_new, file_id, start_date,end_date)
                print(file_id, i, "passed")
            except Exception as e:
                print(f"Error occurred for {file_id}: {e}. Skipping to the next iteration.")
                continue
        else:
            try:
                ldn[file_id] = dailyprofile_data(dl_all, file_id, start_date, end_date)
                print(file_id, i, "passed")
            except Exception as e:
                print(f"Error occurred for {file_id}: {e}. Skipping to the next iteration.")
                continue
    grouped_data,grouped_data_cat,merge_data= merge_highlow_mapp(dfs,ldn, high_low= False)
    excel_file = f'{title}_{year}.xlsx'
    # Create an Excel file with sheets three sheets
    # 1. Mean 96 timeslots for each building type
    # 2. Mean 96 slots for all data
    # 3. Mean 96 slots for each ldn id
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Write each DataFrame to a different worksheet.
        grouped_data.to_excel(writer, sheet_name='Building type')
        grouped_data_cat.to_excel(writer, sheet_name='Mean')
        merge_data.to_excel(writer, sheet_name='All data')
    # return grouped_data,grouped_data_cat,merge_data

# Function to read 2022,2023,2024 files and return dataframes
def readfiles():
    # Define the filenames and sheet names
    files_sheets = {
        'All_2022.xlsx': ['Building type', 'Mean', 'All data'],
        'All_2023.xlsx': ['Building type', 'Mean', 'All data'],
        'All_2024.xlsx': ['Building type', 'Mean', 'All data']
    }
    dfs = {}
    # Loop through each file and sheet name
    for file, sheets in files_sheets.items():
        file_path = os.path.join(load_data, file)
        for sheet in sheets:
            df = pd.read_excel(file_path, sheet_name=sheet)
            dfs[f"{file}-{sheet}"] = df

    df_2022_build = dfs['All_2022.xlsx-Building type']
    df_2022_mean = dfs['All_2022.xlsx-Mean']
    df_2022_all = dfs['All_2022.xlsx-All data']

    df_2023_build = dfs['All_2023.xlsx-Building type']
    df_2023_mean = dfs['All_2023.xlsx-Mean']
    df_2023_all = dfs['All_2023.xlsx-All data']

    df_2024_build = dfs['All_2024.xlsx-Building type']
    df_2024_mean = dfs['All_2024.xlsx-Mean']
    df_2024_all = dfs['All_2024.xlsx-All data']

    return(df_2022_build,df_2022_mean,df_2022_all,
           df_2023_build,df_2023_mean,df_2023_all,df_2024_build,df_2024_mean,df_2024_all)


# Function to read dynamic price csv file
def dyn_price_read():
    df_price_dyn = pd.read_csv(dyn_dir)
    df_price_dyn['Timestamp_UTC'] = pd.to_datetime(df_price_dyn['Timestamp_UTC'], dayfirst=True)
    df_price_dyn['Timestamp_UTC'] = df_price_dyn['Timestamp_UTC'].dt.tz_localize('UTC')
    df_price_dyn['Timestamp_Local'] = df_price_dyn['Timestamp_UTC'].dt.tz_convert('Europe/Zurich')
    df_price_dyn.drop(columns=['Timestamp_UTC'], inplace=True)
    df_price_dyn['Date_time'] = pd.to_datetime(df_price_dyn['Timestamp_Local']).dt.tz_convert(None)
    df_price_dyn = df_price_dyn[['Timestamp_Local', 'Price_Rp_per_kWh','Date_time']]
    return df_price_dyn

# Function to identify peak period for dynamic prices
def peak_ident(start_date,end_date):
    df = dyn_price_read()
    df = df[(df['Date_time'] >= pd.to_datetime(start_date, dayfirst=True)) & (df['Date_time'] <= pd.to_datetime(end_date, dayfirst=True))]
    df['slot'] = df['Date_time'].dt.hour * 4 + df['Date_time'].dt.minute // 15 + 1
    avg_price_per_period = df.groupby('slot')['Price_Rp_per_kWh'].mean().reset_index()
    peak_threshold = avg_price_per_period['Price_Rp_per_kWh'].quantile(0.75)
    avg_price_per_period['period_type'] = avg_price_per_period['Price_Rp_per_kWh'].apply(lambda x: 'peak' if x >= peak_threshold else 'off-peak')
    avg_price_per_period= avg_price_per_period[['slot','period_type']]
    return avg_price_per_period

# Function to calculate energy in case of Ortnetz tariffs
def energy_sum(dl, file_id,mode, start_date, end_date):
    df = dl.load(file_id)
    df = df.dropna(subset=['Timestamp'])
    # Extracting date components
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['Hour'] = df['Timestamp'].dt.hour
    # Calculating active and reactive power
    df['ActiveP'] = df['A+'] - df['A-']
    # Removing timezone information
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    df['Date'] = pd.to_datetime(df['Date'],errors='coerce').dt.date
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], errors='coerce')
    filtered_df['Year'] = filtered_df['Date'].dt.year

    if mode == 'Dyn':
    # Reading dynmaic tariff
        df_dyn= dyn_price_read()
        df_dyn = df_dyn.rename(columns={'Date_time': 'Timestamp'})
        merged_df = pd.merge(filtered_df, df_dyn, how='left', on='Timestamp')
        merged_df['Network_cost'] = (merged_df['ActiveP'] * merged_df['Price_Rp_per_kWh']) / 100

        return_df = merged_df.groupby('Date')[['ActiveP', 'Network_cost']].sum()
        return return_df

    elif mode == 'Tou':
        filtered_df['Condition'] = 'low'
        # Iterate over unique years in the DataFrame
        for year in filtered_df['Year'].unique():
            if year == 2023:
                # High Condition for 2023
                conditions_high = (
                        (filtered_df['Hour'].between(18, 23)) &
                        (filtered_df['Date'].between(pd.to_datetime("2023-10-01"), pd.to_datetime("2023-12-31")))
                )
                filtered_df.loc[conditions_high, 'Condition'] = 'high'
            elif year == 2024:
                # Winter conditions (Jan-April) - High
                conditions_high_winter = (
                        (filtered_df['Hour'].between(18, 23)) &
                        (filtered_df['Date'].between(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-04-30")))
                )
                filtered_df.loc[conditions_high_winter, 'Condition'] = 'high'
                # Summer conditions (May-September) - Low
                conditions_low = (
                        (filtered_df['Hour'].between(13, 17)) &
                        (filtered_df['Date'].between(pd.to_datetime("2024-05-01"), pd.to_datetime("2024-09-30")))
                )
                filtered_df.loc[conditions_low, 'Condition'] = 'low'
                # Summer conditions (May-September) - Medium
                conditions_medium = (
                        (filtered_df['Hour'].between(0, 6)) &
                        (filtered_df['Date'].between(pd.to_datetime("2024-05-01"), pd.to_datetime("2024-09-30")))
                )
                filtered_df.loc[conditions_medium, 'Condition'] = 'medium'
                # Summer conditions (May-September) - High (all other hours)
                conditions_high_summer = (
                        ~(conditions_low | conditions_medium) &  # Exclude low and medium hours
                        (filtered_df['Date'].between(pd.to_datetime("2024-05-01"), pd.to_datetime("2024-09-30")))
                )
                filtered_df.loc[conditions_high_summer, 'Condition'] = 'high'
            # Handle any other years if necessary
            else:
                filtered_df.loc[filtered_df['Year'] == year, 'Condition'] = 'high'
        # Print the final DataFrame
        sum_activep_high =  filtered_df[filtered_df['Condition'] == 'high'].groupby('Date')['ActiveP'].sum()
        sum_activep_medium =  filtered_df[filtered_df['Condition'] == 'medium'].groupby('Date')['ActiveP'].sum()
        sum_activep_low =  filtered_df[filtered_df['Condition'] == 'low'].groupby('Date')['ActiveP'].sum()
        return [sum_activep_high,sum_activep_medium,sum_activep_low]
    elif mode == 'All':
        conditions = (filtered_df['Weekday'].between(0, 4)) & (filtered_df['Hour'].between(7, 19))
        filtered_df['Tariff'] = np.where(conditions, 'High', 'Low')
        high_energy = filtered_df[filtered_df['Tariff'] == 'High'].groupby('Date')['ActiveP'].sum()
        low_energy = filtered_df[filtered_df['Tariff'] == 'Low'].groupby('Date')['ActiveP'].sum()
        return [high_energy,low_energy]

# Function to extract energy for OrtsNetz Tariffs
def extraction(dl_all,dl_new,hdf_new,hdf,mode,start_date,end_date):

    ldn = {}
    for i, file_id in enumerate(hdf):
        if file_id in hdf_new:
            try:
                ldn[file_id] = energy_sum(dl_new, file_id,mode, start_date,end_date )
                print(file_id, i, "passed first")
            except Exception as e:
                print(f"Error occurred for {file_id}: {e}. Skipping to the next iteration.")
                continue
        else:
            try:
                ldn[file_id] = energy_sum(dl_all, file_id,mode,start_date,end_date)
                print(file_id, i, "passed second")
            except Exception as e:
                print(f"Error occurred for {file_id}: {e}. Skipping to the next iteration.")
                continue
    return ldn

# Function to merge energy extracted from ldn ids based on mode(Tou,Dyn,All)
# In case of Dyn Network cost is returned from extraction function along with Active Power
def merge_high_low(ldn,mode):
    dfs = []
    if mode == 'Tou':
        for file_id, series_list in ldn.items():
            non_empty_series = [series for series in series_list if not series.empty]
            if len(non_empty_series) >= 2:
                df = pd.concat(non_empty_series, axis=1)
                # Determine column names based on the number of non-empty series for 'Tou'
                if len(non_empty_series) == 2:
                    col_names = ['high_energy', 'low_energy']
                elif len(non_empty_series) == 3:
                    col_names = ['high_energy', 'medium_energy', 'low_energy']
                else:
                    raise ValueError("Unexpected number of series for mode 'Tou'")
                df.columns = col_names
                df['aim_geraet_ldn'] = file_id
                dfs.append(df)
        result_df = pd.concat(dfs)
        result_df.reset_index(inplace=True)

    elif mode == 'All':
        for file_id, series_list in ldn.items():
            non_empty_series = [series for series in series_list if not series.empty]
            # Now, only include series with more than one column
            non_empty_series = [series for series in non_empty_series if len(series) > 1]
            if len(non_empty_series)>=2:
                df = pd.concat(non_empty_series, axis=1)
                col_names = ['high_energy', 'low_energy']
                # Rename the columns and add file_id
                df.columns = col_names
                df['aim_geraet_ldn'] = file_id
                dfs.append(df)
        result_df = pd.concat(dfs)
        result_df.reset_index(inplace=True)
    elif mode =='Dyn':
        for meter_id, ldn_df in ldn.items():
            # Reset the index to ensure 'Date' is a column
            ldn_df = ldn_df.reset_index()
            # Add the meter id to each DataFrame
            ldn_df['aim_geraet_ldn'] = meter_id
            dfs.append(ldn_df)
            # Concatenate all DataFrames into one
        result_df = pd.concat(dfs, ignore_index=True)
    # Mapping
    df = pd.read_csv(os.path.join(dir, 'mapping.csv'), sep=';')
    df = df[['Anlagenart', 'aim_geraet_ldn', 'Geschäftspartner', 'nis_name']]
    df = df.dropna(subset=['aim_geraet_ldn'])
    merged_df = pd.merge(
        result_df,
        df[['aim_geraet_ldn', 'Geschäftspartner', 'Anlagenart', 'nis_name']],
        on='aim_geraet_ldn',
        how='left'
    )
    merged_df.fillna(0, inplace=True)
    return merged_df

# Function to mapping the tariffs with energy and calculate Energy/Network cost
def mapping_tarif(df, dfs, mode):
    # Unpack the dfs
    df_optin, df_notoptout, df_auto_wout_dir, df_manu = dfs

    # Convert 'Date' column to datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract the year and month from the 'Date' column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Group the relevant sets
    optin_set = set(df_optin['Geschäftspartner'])
    notoptout_set = set(df_notoptout['Geschäftspartner'])
    auto_wout_dir_set = set(df_auto_wout_dir['Geschäftspartner'])
    manu_set = set(df_manu['Geschäftspartner'])

    # Define the internal functions
    def determine_keyword(x):
        if x in optin_set:
            return f'Optin {mode}'
        elif x in notoptout_set:
            return f'Not opt out {mode}'
        return ''

    def determine_type(x):
        if x in auto_wout_dir_set:
            return 'Auto'
        elif x in manu_set:
            return 'Manual'
        return ''

    # Apply mappings for 'Category' and 'Control'
    df['Category'] = df['Geschäftspartner'].apply(determine_keyword)
    df['Control'] = df['Geschäftspartner'].apply(determine_type)

    # Define the default values for different modes with month-based tariff distinction for 2024
    default_values = {
        'Dyn': {2023: (10.8, 2.9205, 8), 2024: (18.1, 4.41, 8)},
        'Dyn_wd_flex': {2023: (10.35, 2.9205, 8), 2024: (17.5, 4.41, 8)},
        'Tou': {
            2023: (10.8, 9.3, 0, 4.3, 2.9205, 8),
            '2024_1': (18.1, 10, 0, 5.55, 4.41, 8),  # Jan to Apr
            '2024_2': (17.5, 8.2, 6.9, 3.7, 4.41, 8)  # May onwards
        },
        'Tou_wd_flex': {
            2023: (10.35, 4.7, 0, 4.7, 2.9205, 8),
            '2024_1': (17.5, 10, 0, 5.55, 4.41, 8),  # Jan to Apr
            '2024_2': (17.5, 8.2, 6.9, 3.7, 4.41, 8)  # May onwards
        },
        'All': {2023: (11.45, 10.35, 7.6, 4.7, 2.9205, 8), 2024: (19, 17.5, 6.9, 6.9, 4.41, 8)}
    }

    # Function to assign costs based on mode, year, and month
    def assign_costs(row):
        year = row['Year']
        month = row['Month']
        try:
            # Use month-based lookup for Tou and Tou_wd_flex in 2024
            if mode in ['Tou', 'Tou_wd_flex'] and year == 2024:
                costs = default_values[mode]['2024_1'] if month in [1, 2, 3, 4] else default_values[mode]['2024_2']
            else:
                # General case for all other years and modes
                costs = default_values[mode][year]

            if mode == 'Tou' or mode == 'Tou_wd_flex':
                row['Energy_cost'], row['Grid_high'], row['Grid_med'], row['Grid_low'], row['othercost'], row[
                    'baseprice'] = costs
            elif mode == 'Dyn' or mode == 'Dyn_wd_flex':
                row['Energy_cost'], row['othercost'], row['baseprice'] = costs
            elif mode == 'All':
                row['Energy_high'], row['Energy_low'], row['Grid_high'], row['Grid_low'], row['othercost'], row[
                    'baseprice'] = costs
        except KeyError:
            raise ValueError(f"No tariff data available for the year {year} in mode '{mode}'.")
        return row

    # Apply cost assignment function
    df = df.apply(assign_costs, axis=1)

    # Calculate 'ActiveP' and cost columns based on mode
    if mode == 'Tou' or mode == 'Tou_wd_flex':
        if 'medium_energy' not in df.columns:
            df['medium_energy'] = 0
        df['ActiveP'] = df['high_energy'] + df['low_energy'] + df['medium_energy']
        df['Energy_cost'] = df['ActiveP'] * df['Energy_cost'] * 0.01
        df['Network_high'] = df['high_energy'] * df['Grid_high'] * 0.01
        df['Network_low'] = df['low_energy'] * df['Grid_low'] * 0.01
        df['Network_med'] = df['medium_energy'] * df['Grid_med'] * 0.01
        df['Network_cost'] = df['Network_high'] + df['Network_low'] + df['Network_med']
    elif mode == 'All':
        df['ActiveP'] = df['high_energy'] + df['low_energy']
        df['Network_high'] = df['high_energy'] * df['Grid_high'] * 0.01
        df['Network_low'] = df['low_energy'] * df['Grid_low'] * 0.01
        df['Energy_low_cost'] = df['low_energy'] * df['Energy_low'] * 0.01
        df['Energy_high_cost'] = df['high_energy'] * df['Energy_high'] * 0.01
        df['Network_cost'] = df['Network_high'] + df['Network_low']
        df['Energy_cost'] = df['Energy_high_cost'] + df['Energy_low_cost']
    elif mode == 'Dyn' or mode == 'Dyn_wd_flex':
        df['Energy_cost'] = df['ActiveP'] * df['Energy_cost'] * 0.01

    # Other and basic cost calculations
    df['Other_cost'] = df['ActiveP'] * df['othercost'] * 0.01
    df['Basic_cost'] = df.apply(lambda row: len(str(row['aim_geraet_ldn']).split(',')) * row['baseprice'], axis=1)
    df['Total_cost'] = df['Energy_cost'] + df['Other_cost'] + df['Network_cost'] + df['Basic_cost']

    # Select columns for output based on mode
    if mode == 'Tou':
        df = df[['Date', 'Geschäftspartner', 'aim_geraet_ldn', 'nis_name', 'Category', 'Control', 'Anlagenart',
                 'high_energy', 'medium_energy', 'low_energy', 'ActiveP', 'Energy_cost', 'Network_high', 'Network_med',
                 'Network_low', 'Network_cost', 'Other_cost', 'Basic_cost', 'Total_cost']]
    elif mode == 'Dyn':
        df = df[
            ['Date', 'Geschäftspartner', 'aim_geraet_ldn', 'nis_name', 'Category', 'Control', 'Anlagenart', 'ActiveP',
             'Energy_cost', 'Network_cost', 'Other_cost', 'Basic_cost', 'Total_cost']]

    # Save to Excel with year dynamically
    df.to_excel(f"{mode}_tariff.xlsx", index=False)
    return df


# Function to merge all the Tariffs from TOU,TOU_Flex,DYN,DYN_Flex,Regular Tariffs together
def merging(df_all,df_tou, df_dyn, df_tou_flex, df_dyn_flex ):
    df_tou = df_tou[['Date','aim_geraet_ldn','Geschäftspartner','Energy_cost','Network_cost','Basic_cost']]
    df_dyn = df_dyn[['Date','aim_geraet_ldn','Geschäftspartner','Energy_cost','Network_cost','Basic_cost']]
    df_tou_flex = df_tou_flex[['Date','aim_geraet_ldn','Geschäftspartner','Energy_cost','Network_cost','Basic_cost']]
    df_dyn_flex = df_dyn_flex[['Date','aim_geraet_ldn','Geschäftspartner','Energy_cost','Network_cost','Basic_cost']]
    df_all = df_all[df_all['Category'].notna() & (df_all['Category'] != '')]
    agg_dict_all = {
        'aim_geraet_ldn': ','.join,
        'nis_name': ','.join,
        'Anlagenart': ','.join,
        'Energy_cost': 'sum',
        'Network_cost': 'sum',
        'Basic_cost': 'sum'
    }
    agg_dict = {
        'aim_geraet_ldn': ','.join,
        'Energy_cost': 'sum',
        'Network_cost': 'sum',
        'Basic_cost': 'sum'
    }
    df_all['aim_geraet_ldn'] = df_all['aim_geraet_ldn'].astype(str)
    df_all['nis_name'] = df_all['nis_name'].astype(str)
    df_all['Anlagenart'] = df_all['Anlagenart'].astype(str)


    df_tou['aim_geraet_ldn'] = df_tou['aim_geraet_ldn'].astype(str)
    df_dyn['aim_geraet_ldn'] = df_dyn['aim_geraet_ldn'].astype(str)
    df_tou_flex['aim_geraet_ldn'] = df_tou_flex['aim_geraet_ldn'].astype(str)
    df_dyn_flex['aim_geraet_ldn'] = df_dyn_flex['aim_geraet_ldn'].astype(str)

    df_tou = df_tou.groupby(['Geschäftspartner','Date']).agg(agg_dict).reset_index()
    df_dyn = df_dyn.groupby(['Geschäftspartner', 'Date']).agg(agg_dict).reset_index()
    df_tou_flex = df_tou_flex.groupby(['Geschäftspartner','Date']).agg(agg_dict).reset_index()
    df_dyn_flex = df_dyn_flex.groupby(['Geschäftspartner', 'Date']).agg(agg_dict).reset_index()
    df_all = df_all.groupby(['Geschäftspartner', 'Date']).agg(agg_dict_all).reset_index()

    merged_df = pd.merge(df_all, df_tou, on=['Geschäftspartner', 'Date'], how='left', suffixes=('', '_Tou'))
    merged_df = pd.merge(merged_df, df_tou_flex, on=['Geschäftspartner', 'Date'], how='left', suffixes=('', '_Tou_flex'))

    merged_df = pd.merge(merged_df, df_dyn, on=['Geschäftspartner', 'Date'], how='left', suffixes=('', '_Dyn'))
    merged_df = pd.merge(merged_df, df_dyn_flex, on=['Geschäftspartner', 'Date'], how='left', suffixes=('', '_Dyn_flex'))


    def get_non_zero_network_cost(row):
        # Handle NaNs and zeros by checking both conditions
        if pd.notnull(row['Network_cost_Dyn']) and row['Network_cost_Dyn'] != 0:
            return row['Network_cost_Dyn']
        elif pd.notnull(row['Network_cost_Tou']) and row['Network_cost_Tou'] != 0:
            return row['Network_cost_Tou']
        elif pd.notnull(row['Network_cost_Dyn_flex']) and row['Network_cost_Dyn_flex'] != 0:
            return row['Network_cost_Dyn_flex']
        elif pd.notnull(row['Network_cost_Tou_flex']) and row['Network_cost_Tou_flex'] != 0:
            return row['Network_cost_Tou_flex']
        else:
            return np.nan  # Return NaN if no valid values found

    def get_non_zero_energy_cost(row):
        if pd.notnull(row['Energy_cost_Dyn']) and row['Energy_cost_Dyn'] != 0:
            return row['Energy_cost_Dyn']
        elif pd.notnull(row['Energy_cost_Tou']) and row['Energy_cost_Tou'] != 0:
            return row['Energy_cost_Tou']
        elif pd.notnull(row['Energy_cost_Dyn_flex']) and row['Energy_cost_Dyn_flex'] != 0:
            return row['Energy_cost_Dyn_flex']
        elif pd.notnull(row['Energy_cost_Tou_flex']) and row['Energy_cost_Tou_flex'] != 0:
            return row['Energy_cost_Tou_flex']
        else:
            return np.nan  # Return NaN if no valid values found

    # Apply the functions to the dataframe
    merged_df['OrtsNetz_Network_cost'] = merged_df.apply(get_non_zero_network_cost, axis=1)
    merged_df['OrtsNetz_Energy_cost'] = merged_df.apply(get_non_zero_energy_cost, axis=1)

    merged_df['Energy_saving'] = np.where(merged_df['Energy_cost'] - merged_df['OrtsNetz_Energy_cost'] >= 0,merged_df['Energy_cost'] - merged_df['OrtsNetz_Energy_cost'],0)
    merged_df['Network_saving'] = np.where(merged_df['Network_cost'] - merged_df['OrtsNetz_Network_cost'] >= 0,merged_df['Network_cost'] - merged_df['OrtsNetz_Network_cost'],0)

    df_anlys = merged_df[['Date','Geschäftspartner','aim_geraet_ldn','Energy_cost','OrtsNetz_Energy_cost', 'Energy_saving','Network_cost','OrtsNetz_Network_cost','Network_saving','Basic_cost']]
    df_anlys.fillna(0)
    return df_anlys

# To analyse the enrollment data sheet
def process_enrollment_data():
    dfs_customer_data = pd.read_excel(enrollment_dir)
    # energy&cost_2022 sheet generated for comparing whether data B.P.Numbers
    dfs_energy_costs = pd.read_excel('D:\\master thesis\\energy&costs_2022.xlsx',sheet_name='Main')

    # Read data from Excel files
    df_main = dfs_energy_costs[dfs_energy_costs['Geschäftspartner'] != 0]

    # Calculate Total_cost for df_BP_of_interest and df_main (to shift to customers energy sheet)
    df_main['Total_energy'] = df_main['high_energy'] + df_main['low_energy']
    df_main['Total_cost'] = df_main['Total_high_cost'] + df_main['Total_low_cost']+df_main['Total_baseprice']

    #All customers
    df_all = dfs_customer_data
    # Defining customers in Ortsnetz tariff
    df_otnz_brf = dfs_customer_data[(dfs_customer_data['OrtsNetz_Tarif_Brief'] == True) & (dfs_customer_data['abgemeldet_LSG'] == False) & (dfs_customer_data['Abmeldung_OrtsNetz_Tarif_v_LSG_Brief'] == False)].reset_index()
    filtered_otnz_brf = df_main[df_main['Geschäftspartner'].isin(df_otnz_brf['Geschäftspartner'])]

    # Defining customers in Ortznetz tariff excluding direct control
    df_otnz_brf_idr = df_otnz_brf[(df_otnz_brf['Tarif_Einheits_Flex_Brief'] == False) & (df_otnz_brf['Tarif_Einheits_ohne_Flex_Brief'] == False)]
    filtered_otnz_brf_idr = df_main[df_main['Geschäftspartner'].isin(df_otnz_brf_idr['Geschäftspartner'])]

    # Defininog Opt in customers
    df_optin = df_otnz_brf_idr[(df_otnz_brf_idr['online'] == True)]
    filtered_optin = df_main[df_main['Geschäftspartner'].isin(df_optin['Geschäftspartner'])]

    # Defining Not Opt out customers
    df_notoptout = df_otnz_brf_idr[(df_otnz_brf_idr['online'] == False)]
    filtered_not_opt_out = df_main[df_main['Geschäftspartner'].isin(df_notoptout['Geschäftspartner'])]

    # customers with Devices
    df_auto = dfs_customer_data[(dfs_customer_data['bekommt_LSG'] == True) & (dfs_customer_data['LSG_nicht_erreichbar'] == False) & (dfs_customer_data['LSG_nicht_in_TS'] == False) & (dfs_customer_data['abgemeldet_LSG'] == False) & (dfs_customer_data['Abmeldung_OrtsNetz_Tarif_v_LSG_Brief'] == False)]
    filtered_auto = df_main[df_main['Geschäftspartner'].isin(df_auto['Geschäftspartner'])]
    # customers with Devices & Direct control
    df_direct = df_auto[(dfs_customer_data['Tarif_Einheits_Flex_Brief'] == True)]
    filtered_dir = df_main[df_main['Geschäftspartner'].isin(df_direct['Geschäftspartner'])]

    # customers with Devices & without direct control
    df_auto_wout_dir = df_auto[(dfs_customer_data['Tarif_Einheits_Flex_Brief'] == False)]
    filtered_auto_wout_dir = df_main[df_main['Geschäftspartner'].isin(df_auto_wout_dir['Geschäftspartner'])]

    # customers with Manual control
    df_manu = df_otnz_brf_idr[~df_otnz_brf_idr['Geschäftspartner'].isin(df_auto_wout_dir['Geschäftspartner'])]
    filtered_manu = df_main[df_main['Geschäftspartner'].isin(df_manu['Geschäftspartner'])]


    df_optin_tou = df_optin[(df_optin['Tarif_TOU_ohne_Flex_Brief'] == True)|(df_optin['Tarif_TOU_Flex_Brief'] == True)]
    filtered_optin_tou = df_main[df_main['Geschäftspartner'].isin(df_optin_tou['Geschäftspartner'])]

    df_optin_dyn = df_optin[(df_optin['Tarif_dynamic_Flex_Brief'] == True)|(df_optin['Tarif_dynamic_ohne_Flex_Brief'] == True)]
    filtered_optin_dyn = df_main[df_main['Geschäftspartner'].isin(df_optin_dyn['Geschäftspartner'])]


    # Opt in customers with Auto and Manual Load control
    df_optin_auto = df_optin[df_optin['Geschäftspartner'].isin(df_auto_wout_dir['Geschäftspartner'])]
    filtered_optin_auto = df_main[df_main['Geschäftspartner'].isin(df_optin_auto['Geschäftspartner'])]


    df_optin_manu = df_optin[df_optin['Geschäftspartner'].isin(df_manu['Geschäftspartner'])]
    filtered_optin_manu = df_main[df_main['Geschäftspartner'].isin(df_optin_manu['Geschäftspartner'])]

    # Not Opt out customers with Auto and Manual Load control

    df_notoptout_tou = df_notoptout[(df_notoptout['Tarif_TOU_ohne_Flex_Brief'] == True)|(df_notoptout['Tarif_TOU_Flex_Brief'] == True)]
    filtered_notoptout_tou = df_main[df_main['Geschäftspartner'].isin(df_notoptout_tou['Geschäftspartner'])]

    df_notoptout_dyn = df_notoptout[(df_notoptout['Tarif_dynamic_Flex_Brief'] == True)|(df_notoptout['Tarif_dynamic_ohne_Flex_Brief'] == True)]
    filtered_notoptout_dyn = df_main[df_main['Geschäftspartner'].isin(df_notoptout_dyn['Geschäftspartner'])]

    df_notoptout_auto = df_notoptout[df_notoptout['Geschäftspartner'].isin(df_auto_wout_dir['Geschäftspartner'])]
    filtered_not_optout_auto = df_main[df_main['Geschäftspartner'].isin(df_notoptout_auto['Geschäftspartner'])]
    df_notoptout_manu = df_notoptout[df_notoptout['Geschäftspartner'].isin(df_manu['Geschäftspartner'])]
    filtered_not_optout_manu = df_main[df_main['Geschäftspartner'].isin(df_notoptout_manu['Geschäftspartner'])]

    # customers of interest
    df_BP_of_interest  = df_otnz_brf_idr
    filtered_df_int = filtered_otnz_brf_idr
    return df_all,df_main, df_BP_of_interest, filtered_df_int, df_optin,df_optin_dyn,df_optin_tou, df_notoptout,df_notoptout_dyn,df_notoptout_tou,filtered_optin, filtered_not_opt_out, df_auto_wout_dir,filtered_auto_wout_dir,df_manu, filtered_manu, filtered_optin_auto,filtered_optin_manu,filtered_not_optout_auto,filtered_not_optout_manu

# Mann whitney U test :
# Function to calculate lower_energy_pct
def calculate_low_energy_pct(df, group_col):
    energy_total = df.groupby(group_col)[['high_energy', 'low_energy']].sum().reset_index()
    energy_total['Total_energy'] = energy_total['high_energy'] + energy_total['low_energy']
    energy_total['low_energy_pct'] = energy_total['low_energy'] / energy_total['Total_energy'] * 100
    return energy_total

# Function to perform test:
def mannwhit(dfA, dfB,dfC =None, name_A='Group A', name_B='Group B'):
    # Ensure that the necessary columns exist in both dataframes
    if 'low_energy_pct' not in dfA.columns or 'low_energy_pct' not in dfB.columns:
        raise ValueError("Both dataframes must have a 'low_energy_pct' column")

    if 'Geschäftspartner' not in dfA.columns or 'Geschäftspartner' not in dfB.columns:
        raise ValueError("Both dataframes must have a 'Geschäftspartner' column")

    # Replace NaN values with 0 in both dataframes
    dfA['low_energy_pct'] = dfA['low_energy_pct'].fillna(0)
    dfB['low_energy_pct'] = dfB['low_energy_pct'].fillna(0)

    # Remove rows in dfB where 'Geschäftspartner' matches any value in dfA['Geschäftspartner']
    dfB_filtered = dfB[~dfB['Geschäftspartner'].isin(dfA['Geschäftspartner'])]
    if dfC is not None:
        dfB_filtered = dfB[~dfB['Geschäftspartner'].isin(dfC['Geschäftspartner'])]

    # Perform a two-sided Mann-Whitney U test
    stat, p_two_sided = mannwhitneyu(dfA['low_energy_pct'], dfB_filtered['low_energy_pct'], alternative='two-sided',
                                     nan_policy='omit')

    # Print the p-value
    print(f'Two-sided P-value for {name_A} vs. {name_B}:', p_two_sided)

    # Define significance level
    alpha = 0.05

    # Interpret the two-sided p-value
    if p_two_sided < alpha:
        print(f"The Low Tariff energy consumption differs significantly between {name_A} and {name_B}.")

        # Perform one-sided Mann-Whitney U tests to determine direction
        stat, p_less = mannwhitneyu(dfA['low_energy_pct'], dfB_filtered['low_energy_pct'], alternative='less',
                                    nan_policy='omit')
        stat, p_greater = mannwhitneyu(dfA['low_energy_pct'], dfB_filtered['low_energy_pct'], alternative='greater',
                                       nan_policy='omit')

        # Print one-sided p-values
        print(f'One-sided P-value (less) for {name_A} vs. {name_B}:', p_less)
        print(f'One-sided P-value (greater) for {name_A} vs. {name_B}:', p_greater)

        if p_less < alpha:
            print(f"The Low Tariff energy consumption of {name_A} is significantly less than that of {name_B}.\n")
        elif p_greater < alpha:
            print(f"The Low Tariff energy consumption of {name_A} is significantly greater than that of {name_B}.\n")
    else:
        print(f"There is no significant difference in the Low Tariff energy consumption between {name_A} and {name_B}.\n")

if __name__ == "__main__":

    # importing dataframes from plot.py (number and type of customers as data_frame)
    df_all,df_main, df_BP_of_interest, filtered_df_int, df_optin,df_optin_dyn,df_optin_tou, df_notoptout,df_notoptout_dyn,df_notoptout_tou,filtered_optin, filtered_not_opt_out, df_auto_wout_dir,filtered_auto_wout_dir,df_manu, filtered_manu, filtered_optin_auto,filtered_optin_manu,filtered_not_optout_auto,filtered_not_optout_manu = process_enrollment_data()
    # reading smartmeter data
    dl_all,dl_new, hdf_old,hdf_new,hdf = read()
    '''
    # Logic to extract 2022 data for plotting and analysis
    ldn = energy(dl_all,dl_new,hdf_new,hdf,start_date = '2022-01-01',end_date = '2022-12-31')
    dfs = [df_optin_dyn, df_optin_tou, df_notoptout_dyn, df_notoptout_tou, df_auto_wout_dir, df_manu] # Used for mapping the Ldn data with Enrollment data
    df_merg = merge_highlow_mapp(dfs ,ldn, high_low= True)
    df_tarif = tariffs_regular(df_merg,year=2022)

    # above func also creates a file energy&costs_{year}.xlsx which is used as base file for plotting

    # For identifying and removing specific Opernds
    flag = operncode()
    hdf_wo_ornd_set = set(hdf) - set(flag)
    hdf_wo_ornd = list(hdf_wo_ornd_set)

    # Extracting Quarter-hourly Averaged Data for each ldn_id, will create an excel file with three sheets as "{title}_{year}.xlsx"
    dfs = [df_optin_dyn,df_optin_tou,df_notoptout_dyn,df_notoptout_tou,df_auto_wout_dir,df_manu] # For mapping
    # For Winter Period
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2022, title='Averaged_Energy_Data', start_date='2022-01-01', end_date='2022-02-28')
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2023, title='Averaged_Energy_Data', start_date='2023-01-01', end_date='2024-02-28')
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2024, title='Averaged_Energy_Data', start_date='2023-01-01', end_date='2024-02-28')


    # For Transition Period
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2022, title='Averaged_Energy_Data', start_date='2022-03-01', end_date='2022-04-30')
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2023, title='Averaged_Energy_Data', start_date='2023-03-01', end_date='2024-04-30')
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2024, title='Averaged_Energy_Data', start_date='2023-03-01', end_date='2024-04-30')

    # For Summer Period
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2022, title='Averaged_Energy_Data', start_date='2022-05-01', end_date='2022-06-28')
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2023, title='Averaged_Energy_Data', start_date='2023-05-01', end_date='2024-06-28')
    dailyprofile(dfs, dl_all, dl_new, hdf_wo_ornd, hdf_new, year=2024, title='Averaged_Energy_Data', start_date='2023-05-01', end_date='2024-06-28')
   
    ##### For Tariff extraction and analysis ####
    # List of ldn_ids
    optin_dyn = map_filter(df_optin_dyn)
    notoptout_dyn = map_filter(df_notoptout_dyn)
    optin_tou = map_filter(df_optin_tou)
    notoptout_tou = map_filter(df_notoptout_tou)

    Tou = optin_tou + notoptout_tou
    Dyn = optin_dyn + notoptout_dyn

    auto = map_filter(df_auto_wout_dir)
    manu = map_filter(df_manu)

    Tou_with_flex = list(set(Tou) & set(auto))
    dyn_with_flex = list(set(Dyn) & set(auto))

    Tou_wout_flex = list(set(Tou) & set(manu))
    dyn_wout_flex = list(set(Dyn) & set(manu))
    
    dfs = [df_optin, df_notoptout, df_auto_wout_dir, df_manu]
    # TOU energy extraction
    ldn= extraction(dl_all, dl_new,hdf_new,Tou_wout_flex,mode = 'Tou', start_date ='2023-12-01', end_date ='2024-05-31' )
    df_ldn = merge_high_low(ldn, mode = 'Tou')
    df_Tou = mapping_tarif(df_ldn, dfs, mode = 'Tou')

    # Dyn energy extraction
    ldn= extraction(dl_all, dl_new,hdf_new,dyn_wout_flex,mode = 'Dyn', start_date ='2023-12-01', end_date ='2024-05-31' )
    df_ldn = merge_high_low(ldn, mode = 'Dyn')
    df_Dyn = mapping_tarif(df_ldn, dfs, mode = 'Dyn')

    # Dyn energy extraction with flexibility
    ldn = extraction(dl_all, dl_new, hdf_new, dyn_with_flex, mode='Dyn', start_date='2023-12-01', end_date='2024-05-31')
    df_ldn = merge_high_low(ldn, mode='Dyn')
    df_Dyn_flex = mapping_tarif(df_ldn, dfs, mode='Dyn_wd_flex')

    # TOU energy extraction with flexibility
    ldn = extraction(dl_all, dl_new, hdf_new, Tou_with_flex, mode='Tou', start_date='2023-12-01', end_date='2024-05-31')
    df_ldn = merge_high_low(ldn, mode='Tou')
    df_Tou_flex = mapping_tarif(df_ldn, dfs, mode='Tou_wd_flex')

    # All customers extraction based on Normal tariffs
    ldn= extraction(dl_all, dl_new,hdf_new,hdf,mode = 'All', start_date ='2023-12-01', end_date ='2024-05-31')
    df_ldn = merge_high_low(ldn, mode = 'All')
    df_reg = mapping_tarif(df_ldn, dfs, mode = 'All')
    '''

    # This will create also create a new Excel sheet with comparison of Ortsnetz and Regular Tariffs used for plotting
    df_merg_tarf = merging(df_reg,df_Tou,df_Dyn,df_tou_flex, df_dyn_flex)
    dfs = [df_optin_dyn, df_optin_tou, df_notoptout_dyn, df_notoptout_tou, df_auto_wout_dir, df_manu]
    df_map_tarf = mapping(df_merg_tarf,dfs)
    df_map_tarf.to_excel('Tariffs_comparison_mapped.xlsx') # create an Excel sheet
    '''
    # Function to perform test for 2022 data
    filtered_df_optin = calculate_low_energy_pct(filtered_optin, 'Geschäftspartner')
    filtered_df_notoptout = calculate_low_energy_pct(filtered_not_opt_out, 'Geschäftspartner')
    df_cust_interest = calculate_low_energy_pct(filtered_df_int, 'Geschäftspartner')
    df_main = calculate_low_energy_pct(df_main, 'Geschäftspartner')

    # Test pairs
    mannwhit(filtered_df_optin, filtered_df_notoptout, name_A='Opt in Consumers', name_B='Not opt out consumers')
    mannwhit(filtered_df_optin, df_main, filtered_df_notoptout, name_A='Opt in Consumers',name_B='All Consumers excluding opt in consumers and not opt out consumers')
    mannwhit(filtered_df_notoptout, df_main, filtered_df_optin, name_A='Not opt out consumers', name_B='All Consumers excluding not opt out consumers and opt in consumers')
    mannwhit(df_cust_interest, df_main, name_A='Consumers of interest', name_B='All Consumers excluding consumers of interest')
    '''
import h5py
import numpy as np
import pandas as pd
import os
import re
import csv
import itertools
from Consumer_energy import read,map_filter,operncode,mapping
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Consumer_energy import process_enrollment_data as process
from itertools import product


basdir = 'D:\\master thesis\\Smart meter data\\smartmeter-main\\smartmeter-main\\smartmeter\\Load Data\\comparison_01_Jan_28_Feb\\removing specific operends'



def process_and_analyze_data(base_dir,year =2022):
    # Load only the necessary columns from the Excel files
    file_2022 = os.path.join(base_dir, 'All_2022.xlsx')
    file_2023 = os.path.join(base_dir, 'All_2023.xlsx')
    file_2024 = os.path.join(base_dir, 'All_2024.xlsx')

    # Load the data from Excel files
    df1 = pd.read_excel(file_2022, sheet_name='All data')
    df2 = pd.read_excel(file_2023, sheet_name='All data')
    df3 = pd.read_excel(file_2024, sheet_name='All data')

    # Merge the dataframes in one step
    df2_trimmed = df2[['slot', 'aim_geraet_ldn', 'ActiveP']]
    df3_trimmed = df3[['slot', 'aim_geraet_ldn', 'ActiveP']]

    # Rename the ActiveP column to differentiate after merge
    df2_trimmed = df2_trimmed.rename(columns={'ActiveP': 'ActiveP_2023'})
    df3_trimmed = df3_trimmed.rename(columns={'ActiveP': 'ActiveP_2024'})

    # Merge df1 with the trimmed df2 on 'aim_geraet_ldn' and 'slot'
    df_merged = pd.merge(df1, df2_trimmed, on=['aim_geraet_ldn', 'slot'], how='left')

    # Merge the result with the trimmed df3 on 'aim_geraet_ldn' and 'slot'
    df_merged = pd.merge(df_merged, df3_trimmed, on=['aim_geraet_ldn', 'slot'], how='left')

    # Rename column after merging (regardless of whether df_merged was None or not)
    df_merged = df_merged.rename(columns={'ActiveP': 'ActiveP_2022'})

    # Logic to identify the meterids which are zero in one period and then added in another
    grouped_sum = df_merged.groupby('aim_geraet_ldn')[['ActiveP_2022', 'ActiveP_2023', 'ActiveP_2024']].sum()
    removed_values = grouped_sum.loc[(grouped_sum[['ActiveP_2022', 'ActiveP_2023', 'ActiveP_2024']] == 0).any(axis=1)]
    removed_values_list = removed_values.index.to_list()


    # Filter merged dataframe and calculate percentage changes
    filtered_df = df_merged[~df_merged['aim_geraet_ldn'].isin(removed_values_list) &
                            (df_merged['Control'].isnull() | df_merged['Control'].eq(''))].reset_index()
    def normal_2022(filtered_df):
        # Avoid SettingWithCopyWarning by using assign for new columns
        filtered_df = filtered_df.assign(
            Change_2023=(filtered_df['ActiveP_2023'] - filtered_df['ActiveP_2022']) / filtered_df['ActiveP_2022'],
            Change_2024=(filtered_df['ActiveP_2024'] - filtered_df['ActiveP_2022']) / filtered_df['ActiveP_2022'],
            Abs_2023=abs((filtered_df['ActiveP_2024'] - filtered_df['ActiveP_2022'])),
            Abs_2024=abs((filtered_df['ActiveP_2024'] - filtered_df['ActiveP_2022']))
        )
        x= 3 # For max 300% of change

        filtered_df = filtered_df[(filtered_df['Change_2023'].between(-x, x)) &
                                  (filtered_df['Change_2024'].between(-x, x))]

        grouped_df = filtered_df.groupby(['slot', 'Anlagenart']).agg({
            'Change_2023': 'median',
            'Change_2024': 'median','ActiveP_2023': 'sum','ActiveP_2024': 'sum'
        }).reset_index()

        return grouped_df

    def normal_2023(filtered_df):
        filtered_df = filtered_df.assign(
            Change_2024=(filtered_df['ActiveP_2024'] - filtered_df['ActiveP_2023']) / filtered_df['ActiveP_2023'],
            Abs_2024=abs((filtered_df['ActiveP_2024'] - filtered_df['ActiveP_2023']))
        )
        x = 3  # For max 300% of change

        filtered_df = filtered_df[(filtered_df['Change_2024'].between(-x, x))]

        grouped_df = filtered_df.groupby(['slot', 'Anlagenart']).agg({'Change_2024': 'median','ActiveP_2024': 'sum' }).reset_index()

        return grouped_df

    if year == 2022:
        grouped_df = normal_2022(filtered_df)
    else:
        grouped_df= normal_2023(filtered_df)

    # Save the grouped DataFrame to an Excel file
    grouped_df.to_excel(f'normalise_{year}.xlsx', index=False)

    return grouped_df


if __name__ == "__main__":
    merged_temp= temp()
    df_all,df_main, df_BP_of_interest, filtered_df_int, df_optin,df_optin_dyn,df_optin_tou, df_notoptout,df_notoptout_dyn,df_notoptout_tou,filtered_optin, filtered_not_opt_out, df_auto_wout_dir,filtered_auto_wout_dir,df_manu, filtered_manu, filtered_optin_auto,filtered_optin_manu,filtered_not_optout_auto,filtered_not_optout_manu = process()
    optin_dyn = map_filter(df_optin_dyn)
    notoptout_dyn = map_filter(df_notoptout_dyn)
    optin_tou = map_filter(df_optin_tou)
    notoptout_tou = map_filter(df_notoptout_tou)
    auto = map_filter(df_auto_wout_dir)
    manu = map_filter(df_manu)
    Tou = optin_tou + notoptout_tou
    Dyn = optin_dyn + notoptout_dyn
    total = Tou + Dyn
    optin = map_filter(df_optin)
    not_opt_out = map_filter(df_notoptout)

    optin_with_flex = list(set(optin) & set(auto))
    not_opt_out_with_flex = list(set(not_opt_out) & set(auto))

    optin_wout_flex = list(set(optin) & set(manu))
    not_opt_out_wout_flex = list(set(not_opt_out) & set(manu))

    # Directory for consumer energy and All data files
    #df= process_and_analyze_data(basdir, year =2022)

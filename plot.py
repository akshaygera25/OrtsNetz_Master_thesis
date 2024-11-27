import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind, levene, mannwhitneyu,shapiro
from Consumer_energy import process_enrollment_data as process
from Consumer_energy import readfiles, slot_to_time,peak_ident
import os
from Normalise import process_and_analyze_data as normalise_one
from matplotlib.ticker import FuncFormatter

building_types = ['Wohnung', 'Eigentumswohnung', 'Einfamilienhaus']
building_types_eng = ['Apartment', 'Condominium ', 'Single-family house']

building_types_map = {
    'Wohnung': 'Apartment',
    'Eigentumswohnung': 'Condominium',
    'Einfamilienhaus': 'Single-family house'
}
labels = ['All consumers', 'Consumers of Interest', 'Opt-in Consumers', 'Not Opt out Consumers']
dfs_energy_costs = pd.read_excel('D:\\master thesis\\energy&costs_2022.xlsx', sheet_name=None)
dfs_customer_data = pd.read_excel('D:\\master thesis\\Smart meter data\\all_customer_data_2024-06-13_reduced.xlsx')

# For normalisation
load_data = 'D:\\master thesis\\Smart meter data\\smartmeter-main\\smartmeter-main\\smartmeter\\Load Data\\Winter_Dec-Feb'


def calculate_avg_total(df, building_types, col):
    # Calculate for specified building types
    filtered_df_specific = df[df['Anlagenart'].isin(building_types)]
    avg_specific = filtered_df_specific.groupby('Anlagenart')[col].mean()
    # Calculate for "Others"
    filtered_df_others = df[~df['Anlagenart'].isin(building_types)]
    avg_others = filtered_df_others.groupby(lambda x: 'Others')[col].mean()
    return pd.concat([avg_specific, avg_others])

def calculate_low_energy_pct(df,col,filter=True):
    energy_total = df.groupby(col)[['high_energy', 'low_energy']].sum().reset_index()
    energy_total['Total_energy'] = energy_total['high_energy'] + energy_total['low_energy']
    energy_total['low_energy_pct'] = energy_total['low_energy'] / energy_total['Total_energy'] * 100
    energy_total['high_energy_pct'] = energy_total['high_energy'] / energy_total['Total_energy'] * 100
    # Filter out low_energy_pct values outside 0-100%
    if filter == True:
        energy_total = energy_total[(energy_total['low_energy_pct'] >= 0) & (energy_total['low_energy_pct'] <= 100)]
        energy_total = energy_total[(energy_total['high_energy_pct'] >= 0) & (energy_total['high_energy_pct'] <= 100)]
    return energy_total

def plot_building_metric(dfs,labels, col=None):
    if col == 'Total_cost':
        ylabel = 'Average Cost Paid (CHF)'
        output_file = 'avg_cost_by_building_type'
    elif col == 'Total_energy':
        ylabel = 'Average Energy Consumed (kWh)'
        output_file = 'avg_energy_by_building_type'
    else:
        raise ValueError("Unknown column name. Please use 'Total_cost' or 'Total_energy'.")

    percentage_lists = []
    for df in dfs:
        averages = calculate_avg_total(df, building_types, col)
        percentage_lists.append(averages)

    # Plotting
    x = range(len(building_types) + 1)  # +1 for "Others"
    width = 0.2  # width of each bar

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, averages in enumerate(percentage_lists):
        offsets = [p + i * width for p in x]
        bars = ax.bar(offsets, averages.values, width=width, label=labels[i])

        # Add labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')
    ax.set_ylabel(ylabel, fontsize = 18)
    plt.yticks(fontsize=18)
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(building_types_eng + ['Others'], rotation=0,fontsize =18)  # Include 'Others'
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'({output_file}.pdf')
    plt.savefig(f'({output_file}.jpeg')
    plt.show()

def plot_percent_builtype(dfs, labels):

    percentage_lists = []
    for df in dfs:
        percentages = {}
        total_rows = df.shape[0]

        if total_rows == 0:
            raise ValueError("One of the DataFrames is empty.")

        for building in building_types:
            count = df['Anlagenart'].str.contains(building, na=False).sum()
            percentages[building] = (count / total_rows) * 100

        count_others = df[~df['Anlagenart'].isin(building_types)].shape[0]
        percentages['Others'] = (count_others / total_rows) * 100
        percentage_lists.append(percentages)

    # Plotting
    x = range(len(building_types)+1) #+1 for others
    width = 0.2  # width of each bar

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, percentages in enumerate(percentage_lists):
        offsets = [p + i * width for p in x]
        bars = ax.bar(offsets, percentages.values(), width=width, label=labels[i])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

    #ax.set_xlabel('Building Types')
    ax.set_ylabel('Percentage', fontsize=18)
    #ax.set_title('Percentage of Each Building Type in different Consumer Category')
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(building_types_eng + ['Others'], rotation=0, fontsize=18)  # Include 'Others'
    plt.yticks(fontsize=18)
    ax.legend()
    plt.savefig('building_vs_conscategy.pdf')
    plt.savefig('building_vs_conscategy.jpeg')
    plt.show()


def calculate_total(df, building_types, col):
    totals = {}
    for building in building_types:
        totals[building] = df[df['Anlagenart'].str.contains(building, na=False)][col].sum()
    totals['Others'] = df[~df['Anlagenart'].isin(building_types)][col].sum()
    return totals


def plot_building_total(dfs, labels, col):
    # Columns configuration
    if col == 'Total_cost':
        ylabel = 'Total Cost Paid (Thousands of CHF)'
        title = 'Total Cost Paid by Building Types'
        output_file = 'total_cost_by_building_type.pdf'
        conversion_factor = 1 / 1000  # Convert CHF to thousands of CHF
    elif col == 'Total_energy':
        ylabel = 'Total Energy Consumed (MWh)'
        title = 'Total Energy Consumed by Building Types'
        output_file = 'total_energy_by_building_type.pdf'
        conversion_factor = 1 / 1000  # Convert kWh to MWh
    else:
        raise ValueError("Unknown column name. Please use 'Total_cost' or 'Total_energy'.")


    # Building types as specified
    building_types = ['Wohnung', 'Eigentumswohnung', 'Einfamilienhaus']

    total_lists = []
    for df in dfs:
        totals = calculate_total(df, building_types, col)
        total_lists.append(totals)

    # Plotting
    x = range(len(building_types) + 1)  # +1 for "Others"
    width = 0.2  # width of each bar

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, totals in enumerate(total_lists):
        # Apply conversion factor for the given metric
        converted_totals = {k: v * conversion_factor for k, v in totals.items()}

        offsets = [p + i * width for p in x]
        bars = ax.bar(offsets, converted_totals.values(), width=width, label=labels[i])

        # Add labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

    ax.set_xlabel('Building Types')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(building_types + ['Others'], rotation=0)  # Include 'Others'
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Individual distribution
def distribution(df, title=None, col= 'Total_energy'):
    df = df[df[col] != 0]
    # Plot using Matplotlib
    plt.figure(figsize=(12, 8))
    df[col]= df[col]/1000
    n, bins, patches = plt.hist(df[col], bins=30, edgecolor='k', alpha=0.7, color='skyblue')

    # Add labels on top of each bar if the value is greater than zero
    for i in range(len(n)):
        if n[i] > 0:
            plt.text(bins[i] + (bins[i + 1] - bins[i]) / 2, n[i], str(int(n[i])), ha='center', va='bottom')
    plt.ylabel('Number of Consumers',fontsize = 24)

    if col =='Total_cost':
        plt.xlabel('Total Cost (in CHF)',fontsize=24)
        if title == 'All':
            #plt.title('Distribution of cost paid annually by All consumer')
            max_interval = 5000
        elif title == 'Optin':
            #plt.title('Distribution of cost paid annually by Opt-in consumers')
            max_interval = 500
        elif title == 'notoptout':
            #plt.title('Distribution of cost paid annually by Not Opt out consumers')
            max_interval = 500
        else:
            #plt.title('Distribution of cost paid annually by Consumers of interest')
            max_interval = 2000
    else:
        plt.xlabel('Total Energy (in MWh/a)',fontsize =24)
        if title == 'All':
            #plt.title(f'Distribution of total energy consumed annually by All consumers')
            max_interval = 30
        elif title == 'Optin':
            #plt.title('Distribution of total energy consumed annually by Opt-in consumers')
            max_interval = 5
        elif title == 'notoptout':
            #plt.title('Distribution of total energy consumed annually by Not Opt out consumers')
            max_interval = 5
        else:
            #plt.title('Distribution of total energy consumed annually by consumers of interest')
            max_interval = 5

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust x-axis ticks based on title
    max_val = df[col].max()
    plt.xticks(range(0, int(max_val) + 1, max_interval),fontsize=14)
    plt.yticks(fontsize=18)
    plt.savefig(f'Customer_distribution_{title}_{col}.pdf')
    plt.savefig(f'Customer_distribution_{title}_{col}.jpeg')
    plt.show()

# Subplot Distribution
def distribution_all(dfs, labels, col='Total_energy', x_min=None, x_max=None, bins=30):
    # Check if number of DataFrames matches the number of labels
    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames must match the number of labels")

    num_plots = len(dfs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(20, 4 * num_plots), sharex=True)
    fig.subplots_adjust(hspace=0.05, wspace=1)

    global_max_value = 0
    max_intervals = {}

    for df, label in zip(dfs, labels):
        df = df[df[col] != 0]

        # Apply x-axis filter if set
        if x_min is not None:
            df = df[df[col] >= x_min]
        if x_max is not None:
            df = df[df[col] <= x_max]

        max_value = df[col].max()
        global_max_value = max(global_max_value, max_value)

        # Adjust maximum interval dynamically based on the data
        max_intervals[label] = (int(max_value) // bins) * bins

    # Define distinct colors for each histogram
    cmap = plt.get_cmap('tab10')

    for idx, (df, label) in enumerate(zip(dfs, labels)):
        if num_plots == 1:
            ax = axes
        else:
            ax = axes[idx]

        df = df[df[col] != 0]

        # Apply x-axis filter if set
        if x_min is not None:
            df = df[df[col] >= x_min]
        if x_max is not None:
            df = df[df[col] <= x_max]

        n, bins, patches = ax.hist(df[col], bins=bins, alpha=0.5, label=label, edgecolor='k', color=cmap(idx))

        # Add labels on top of each bar if the value is greater than zero
        for i in range(len(n)):
            if n[i] > 0:
                ax.text(bins[i] + (bins[i + 1] - bins[i]) / 2, n[i], str(int(n[i])), ha='center', va='bottom')
        ax.tick_params(axis='both', which='major', labelsize=24)

    #fig.suptitle('Distribution of consumers from different categories',fontsize= 30)
    if col == 'Total_cost':
        xlabel = 'Total Cost (in CHF)'
    else:
        xlabel = 'Total Energy (in kWh)'

    # Set the common x-axis label
    fig.text(0.5, 0.04, xlabel, ha='center',fontsize= 24,va='top')

    # Set the common y-axis label in the middle
    fig.text(0.04, 0.5, 'Number of Consumers', va='center', rotation='vertical',fontsize= 24)

    for ax in (axes if num_plots > 1 else [axes]):
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(fontsize=24)

    #plt.tight_layout()
    plt.savefig(f'Customer_distribution_{col}.pdf')
    plt.savefig(f'Customer_distribution_{col}.jpeg')
    plt.show()

# individual category ploting
def plot_energy_ratio_distribution(filtered_df, category='optin'):
    # Group by Anlagenart and calculate energy percentages
    energy_by_building_type = calculate_low_energy_pct(filtered_df, 'Anlagenart')
    energy_by_building_type_low = calculate_avg_total(energy_by_building_type, building_types, 'low_energy_pct')
    energy_by_building_type_high = calculate_avg_total(energy_by_building_type, building_types, 'high_energy_pct')
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    indices = np.arange(len(energy_by_building_type_low))

    # Plot low energy bars
    bars_low = ax.bar(indices, energy_by_building_type_low, bar_width,
                      label=f'Low Energy %({category})')
    # Plot high energy bars on top of the low energy bars
    bars_high = ax.bar(indices, energy_by_building_type_high, bar_width,
                       bottom=energy_by_building_type_low,
                       label=f'High Energy %({category})')

    # Add text labels
    def add_labels(bars, values, bottom_values):
        for bar, value, bottom in zip(bars, values, bottom_values):
            if value > 0:  # Only annotate bars with positive height
                ax.annotate(f'{value:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, bottom + value / 2),
                            xytext=(0, 0),  # Place text in the middle of the bar segment
                            textcoords="offset points",
                            ha='center', va='center', color='black', fontsize=7)

    add_labels(bars_low, energy_by_building_type_low, [0] * len(energy_by_building_type_low))
    add_labels(bars_high, energy_by_building_type_high, energy_by_building_type_low)

    # Add labels and title
    ax.set_xlabel('Building Type (Anlagenart)', fontsize=14, labelpad=20)
    ax.set_ylabel('Percentage of Energy Consumed', fontsize=14, labelpad=20)
    ax.set_title(f'High and Low Energy Consumption by Building Type ({category.capitalize()})', fontsize=16)

    # Reduce font size for x-axis tick labels
    ax.set_xticks(indices)
    ax.set_xticklabels(energy_by_building_type_low.index, rotation=0, ha='right', fontsize=8)  # Adjust font size here
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'energy_distribution_by_building_type_{category}.pdf')
    plt.show()


def plot_energy_ratio_distribution_multiple(dfs, labels):
    num_plots = len(dfs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    for ax, df, label in zip(axes, dfs, labels):
        # Group by Anlagenart and calculate energy percentages
        energy_by_building_type = calculate_low_energy_pct(df, 'Anlagenart')
        energy_by_building_type_low = calculate_avg_total(energy_by_building_type, building_types, 'low_energy_pct')
        energy_by_building_type_high = calculate_avg_total(energy_by_building_type, building_types, 'high_energy_pct')

        bar_width = 0.35
        indices = np.arange(len(energy_by_building_type_low))

        # Plot low energy bars
        bars_low = ax.bar(indices, energy_by_building_type_low, bar_width,
                          label=f'Low Energy %({label})')
        # Plot high energy bars on top of the low energy bars
        bars_high = ax.bar(indices, energy_by_building_type_high, bar_width,
                           bottom=energy_by_building_type_low,
                           label=f'High Energy %({label})')

        # Add text labels
        def add_labels(bars, values, bottom_values):
            for bar, value, bottom in zip(bars, values, bottom_values):
                if value > 0:  # Only annotate bars with positive height
                    ax.annotate(f'{value:.2f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, bottom + value / 2),
                                xytext=(0, 0),  # Place text in the middle of the bar segment
                                textcoords="offset points",
                                ha='center', va='center', color='black', fontsize=7)

        add_labels(bars_low, energy_by_building_type_low, [0] * len(energy_by_building_type_low))
        add_labels(bars_high, energy_by_building_type_high, energy_by_building_type_low)

        # Add labels and title
        ax.set_xlabel('Building Type (Anlagenart)', fontsize=14, labelpad=20)
        ax.set_ylabel('Percentage of Energy Consumed', fontsize=14, labelpad=20)
        ax.set_title(f'High and Low Energy Consumption by Building Type', fontsize=16)

        # Reduce font size for x-axis tick labels
        ax.set_xticks(indices)
        ax.set_xticklabels(energy_by_building_type_low.index, rotation=0, ha='right',
                           fontsize=8)  # Adjust font size here
        ax.legend()

    plt.tight_layout()
    plt.savefig('energy_distribution_by_building_type_multiple.pdf')
    plt.show()


def plot_energy_ratio(dfs, labels):
    # Combine energy percentages for each dataframe
    combined_data = []
    for df, label in zip(dfs, labels):
        energy_by_building_type = calculate_low_energy_pct(df, 'Anlagenart')
        energy_by_building_type_low = calculate_avg_total(energy_by_building_type, building_types + ['Other'],
                                                          'low_energy_pct')
        energy_by_building_type_high = calculate_avg_total(energy_by_building_type, building_types + ['Other'],
                                                           'high_energy_pct')

        # Combine into a single DataFrame
        combined_df = pd.DataFrame({
            'Building_Type': energy_by_building_type_low.index,
            f'Low_Energy_{label}': energy_by_building_type_low.values,
            f'High_Energy_{label}': energy_by_building_type_high.values
        })
        combined_data.append(combined_df)

        # Merge all combined data on Building_Type
    unique_building_types = combined_data[0]['Building_Type'].unique().tolist()
    final_df = combined_data[0]

    for additional_df in combined_data[1:]:
        final_df = final_df.merge(additional_df, on='Building_Type', how='outer')
    final_df = final_df.fillna(0)
    final_df = final_df.set_index('Building_Type').reindex(unique_building_types).reset_index()

    # Plotting
    bar_width = 0.4
    indices = np.arange(len(final_df)) * (bar_width * len(labels) + 0.1)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = sns.color_palette("Paired", len(labels) * 2)

    for i, label in enumerate(labels):
        low_bars = ax.bar(indices + i * bar_width, final_df[f'Low_Energy_{label}'], bar_width,
                          label=f'Low Energy % ({label})', color=colors[i * 2])
        high_bars = ax.bar(indices + i * bar_width, final_df[f'High_Energy_{label}'], bar_width,
                           bottom=final_df[f'Low_Energy_{label}'],
                           label=f'High Energy % ({label})', color=colors[i * 2 + 1])

        # Add text labels
        def add_labels(bars, values, bottom_values):
            for bar, value, bottom in zip(bars, values, bottom_values):
                if value > 0:  # Only annotate bars with positive height
                    ax.annotate(f'{value:.2f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, bottom + value / 2),
                                xytext=(0, 0),  # Place text in the middle of the bar segment
                                textcoords="offset points",
                                ha='center', va='center', color='black', fontsize=9)

        add_labels(low_bars, final_df[f'Low_Energy_{label}'], [0] * len(final_df))
        add_labels(high_bars, final_df[f'High_Energy_{label}'], final_df[f'Low_Energy_{label}'])

    # Add labels and title
    #ax.set_xlabel('Building Type', fontsize=14, labelpad=20)
    ax.set_ylabel('Percentage of Energy Consumed', fontsize=18, labelpad=20)
    #ax.set_title('Energy consumed during High and Low tariffs period by Building Type', fontsize=16)
    plt.yticks(fontsize=18)
    # Reduce font size for x-axis tick labels
    ax.set_xticks(indices + (len(labels) - 1) * bar_width / 2)
    ax.set_xticklabels(building_types_eng + ['Others'], rotation=0, ha='center', fontsize=18)
    ax.legend()
    plt.tight_layout()
    plt.savefig('energy_distribution_by_building_type_combined.pdf')
    plt.savefig('energy_distribution_by_building_type_combined.jpeg')
    plt.show()

def box_plot_low_energy_pct(df, title=None):
    energy_total = df.groupby('Geschäftspartner')[['high_energy', 'low_energy']].sum().reset_index()
    energy_total['Total_energy'] = energy_total['high_energy'] + energy_total['low_energy']
    energy_total['low_energy_pct'] = energy_total['low_energy'] / energy_total['Total_energy'] * 100

    filtered_energy_total = energy_total[(energy_total['low_energy_pct'] >= 0) & (energy_total['low_energy_pct'] <= 100)]
    filtered_energy_total = filtered_energy_total.dropna(subset=['low_energy_pct'])

    unfiltered_energy_total = energy_total.dropna(subset=['low_energy_pct'])

    fig, ax1 = plt.subplots(figsize=(14, 10))

    ax1.boxplot(filtered_energy_total['low_energy_pct'], notch=False, vert=True, patch_artist=True, widths=0.7,
                boxprops=dict(facecolor='lightblue', color='blue'))
    ax1.set_ylabel('Low Energy Percentage (Filtered Data)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'Box Plot distribution of {title}')

    ax2 = ax1.twinx()
    ax2.boxplot(unfiltered_energy_total['low_energy_pct'], notch=False, vert=True, patch_artist=True, widths=0.7,
                positions=[2], boxprops=dict(facecolor='lightgreen', color='green'))
    ax2.set_ylabel('Low Energy Percentage (Unfiltered Data)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    max_y_value_filtered = int(filtered_energy_total['low_energy_pct'].max())
    min_y_value_filtered = int(filtered_energy_total['low_energy_pct'].min())
    max_y_value_unfiltered = int(unfiltered_energy_total['low_energy_pct'].max())
    min_y_value_unfiltered = int(unfiltered_energy_total['low_energy_pct'].min())
    y_ticks_filtered = range(min_y_value_filtered, max_y_value_filtered + 10, 10) if max_y_value_filtered >= 100 or min_y_value_filtered <= 0 else range(0, 100 + 10, 10)
    y_ticks_unfiltered = range(min_y_value_unfiltered, max_y_value_unfiltered + 10, 50) if max_y_value_unfiltered >= 100 or min_y_value_unfiltered <= 0 else range(0, 100 + 10, 10)
    ax1.set_yticks(y_ticks_filtered)
    ax2.set_yticks(y_ticks_unfiltered)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([f'{title}_Filtered', f'{title}_Unfiltered'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines on the y-axis
    plt.tight_layout()
    plt.savefig(f'Box_plot_{title.lower() if title else "geschäftspartner"}.pdf')
    plt.show()


def box_plot_multiple_dfs(dfs, labels, filter=True):

    # Apply calculate_low_energy_pct to each DataFrame
    dfs = [calculate_low_energy_pct(df,'Geschäftspartner',filter) for df in dfs]

    # Add a 'Category' column to each DataFrame
    for df, label in zip(dfs, labels):
        df['Category'] = label
        df.reset_index(drop=True, inplace=True)

    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Plot using Seaborn
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='Category', y='low_energy_pct', data=combined_df, palette='Set2', notch=False, width=0.5)
    plt.xlabel('Consumer Category', fontsize=15)
    plt.ylabel('Percentage Energy Consumed During Low Tariff Time', fontsize=15)

    # Set y-ticks dynamically based on data range
    max_y_value = int(combined_df['low_energy_pct'].max())
    min_y_value = int(combined_df['low_energy_pct'].min())
    range_y_value = max_y_value - min_y_value
    if max_y_value >= 100 or min_y_value <= 0:
        step_size = max(1, range_y_value // 10)
        plt.yticks(list(range(min_y_value, max_y_value + step_size, step_size)))
    else:
        plt.yticks(list(range(0, 110, 10)))

    plt.title('Box Plot of Energy Consumed During Low Tariff Time for Different Consumer Categories', fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filter:
        plt.savefig('Filtered_Box_plot_low_energy_pct_multiple_categories.pdf')
    else:
        plt.savefig('Unfiltered_Box_plot_low_energy_pct_multiple_categories.pdf')
    plt.show()


def remove_outliers(df, col):
    # Define the lower and upper bounds for outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def violin_plot_multiple_dfs(dfs, labels, title=None, col=None, filter=True):
    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames must match the number of labels")

    processed_dfs = []

    # Apply calculate_low_energy_pct if needed
    for i in range(len(dfs)):
        df = dfs[i].copy()  # Work on a copy to prevent modifying the original DataFrame
        if col == 'low_energy_pct':
            df = calculate_low_energy_pct(df, 'Geschäftspartner', filter)
        elif col in ['Total_energy', 'Total_cost']:
            df = remove_outliers(df, col)  # Remove outliers if column is Total_cost or Total_energy
        df['Category'] = labels[i]  # Add 'Category' column before appending
        processed_dfs.append(df.reset_index(drop=True))

    for df in processed_dfs:
        print(df)  # Print out each df to trace computation values

    # Combine all DataFrames into one
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Plot using Seaborn
    plt.figure(figsize=(14, 10))
    sns.set_palette('Set2')
    if filter == False:
        sns.violinplot(x='Category', y=col, data=combined_df, palette='Set2', inner='box', scale='width', legend=False)
    else:
        sns.violinplot(x='Category', y=col, data=combined_df, palette='Set2', inner='box', scale='width', legend=False,inner_kws=dict(box_width=15, whis_width=2, color="0.4"))
    plt.gca().set_xlabel('')
    #plt.xlabel('Consumer Category', fontsize=15)

    # Set y-axis label based on the column
    if col == 'Total_cost':
        plt.ylabel('Total Cost(in CHF)', fontsize=18)
    elif col == 'Total_energy':
        plt.ylabel('Total Energy(in kWh)', fontsize=18)
    elif col == 'low_energy_pct':
        plt.ylabel('Energy consumed during Low Tariff period (in Percentage)', fontsize=18)
    plt.xticks(fontsize=18)
    # Set y-ticks dynamically based on data range
    max_y_value = int(combined_df[col].max())
    min_y_value = int(combined_df[col].min())
    range_y_value = max_y_value - min_y_value

    if max_y_value >= 100 or min_y_value <= 0:
        step_size = max(1, range_y_value // 10)
        plt.yticks(list(range(min_y_value, max_y_value + step_size, step_size)),fontsize=18)  # Convert range to list
    else:
        plt.yticks(list(range(0, 110, 10)), fontsize=18)  # Convert range to list

    #plt.title(f'Violin Plot of {title}' if title else 'Violin Plot', fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if filter and col == 'low_energy_pct':
        plt.savefig(f'filtered_Violin_plot_{title.lower().replace(" ", "_") if title else "multiple_categories"}.pdf')
        plt.savefig(f'filtered_Violin_plot_{title.lower().replace(" ", "_") if title else "multiple_categories"}.jpeg')
    elif col == 'low_energy_pct':
        plt.savefig(f'Unfiltered_Violin_plot_{title.lower().replace(" ", "_") if title else "multiple_categories"}.pdf')
        plt.savefig(f'Unfiltered_Violin_plot_{title.lower().replace(" ", "_") if title else "multiple_categories"}.jpeg')
    else:
        plt.savefig(f'Violin_plot_{title.lower().replace(" ", "_") if title else "multiple_categories"}_{col}.pdf')
        plt.savefig(f'Violin_plot_{title.lower().replace(" ", "_") if title else "multiple_categories"}_{col}.jpeg')
    plt.show()

def process_plot_loadprofile(
    df_2022_all, df_2023_all, df_2024_all, building_types, plot_columns,
    normalised=False, nom_year=2022, tou_dyn_filter=None,
    include_others=True, include_total=True,period='Summer'):

    if period == 'Summer':
        start_date = '2024-05-01'
        end_date = '2024-06-28'
    elif period == 'Winter':
        start_date = '2023-12-01'
        end_date = '2024-02-28'
    elif period == 'Transition':
        start_date = '2024-03-01'
        end_date = '2024-04-30'

    df_norm = normalise_one(load_data, nom_year )

    def normalize(df_norm, grouped_df, year, building_type, nom_year):
        if building_type == 'all':
            df_norm = df_norm[~df_norm['Anlagenart'].isin(['Werkstatt', 'Hausteil'])]
            if nom_year == 2022:
                df_norm = df_norm.groupby(['slot'])[['Change_2024', 'Change_2023']].mean().reset_index()
            elif nom_year == 2023:
                df_norm = df_norm.groupby(['slot'])[['Change_2024']].mean().reset_index()
            merged = pd.merge(grouped_df, df_norm, on=['slot'], how='left')

        elif building_type is None:
            df_norm = df_norm[~df_norm['Anlagenart'].isin(
                ['Wohnung', 'Eigentumswohnung', 'Einfamilienhaus', 'Werkstatt', 'Hausteil'])]
            if nom_year == 2022:
                df_norm = df_norm.groupby(['slot'])[['Change_2024', 'Change_2023']].mean().reset_index()
            elif nom_year == 2023:
                df_norm = df_norm.groupby(['slot'])[['Change_2024']].mean().reset_index()
                print('2023',df_norm)
            merged = pd.merge(grouped_df, df_norm, on=['slot'], how='left')

        else:
            df_norm = df_norm[df_norm['Anlagenart'] == building_type]
            if nom_year == 2022:
                df_norm = df_norm.groupby(['slot'])[['Change_2024', 'Change_2023']].mean().reset_index()
            elif nom_year == 2023:
                df_norm = df_norm.groupby(['slot'])[['Change_2024']].mean().reset_index()
            merged = pd.merge(grouped_df, df_norm, on=['slot'], how='left')

        if year == '2023' and nom_year == 2023:
            pass
        elif year == '2023':
            merged['ActiveP_norm'] = np.where(
                pd.notna(merged['Change_2023']),
                merged['ActiveP'] / (1 + merged['Change_2023']),
                merged['ActiveP']
            )
            merged = merged[['slot', 'ActiveP_norm']]
            merged = merged.rename(columns={'ActiveP_norm': 'ActiveP'})
        elif year == '2024':
            merged['ActiveP_norm'] = np.where(
                pd.notna(merged['Change_2024']),
                merged['ActiveP'] / (1 + merged['Change_2024']),
                merged['ActiveP']
            )
            merged = merged[['slot', 'ActiveP_norm']]
            merged = merged.rename(columns={'ActiveP_norm': 'ActiveP'})

        return merged

    def filter_df(df, category_filter=None, control_filter=None, tou_dyn_filter=None):
        df = df[df['Anlagenart'].notnull()]
        category_cond = df['Category'].str.contains(category_filter, na=False) if category_filter else df['Category'].isnull()
        control_cond = df['Control'].str.contains(control_filter, na=False) if control_filter else df['Control'].isnull()
        tou_dyn_cond = df['Category'].str.contains(tou_dyn_filter, na=False) if tou_dyn_filter else pd.Series([True]*len(df))
        return df[category_cond & control_cond & tou_dyn_cond]

    dataframes_per_year = {
        '2022': {
            'optin': filter_df(df_2022_all, 'Optin', tou_dyn_filter=tou_dyn_filter),
            'notoptout': filter_df(df_2022_all, 'Not opt out', tou_dyn_filter=tou_dyn_filter),
            'optin_auto': filter_df(df_2022_all, 'Optin', 'Auto', tou_dyn_filter),
            'optin_manual': filter_df(df_2022_all, 'Optin', 'Manual', tou_dyn_filter),
            'notoptout_auto': filter_df(df_2022_all, 'Not opt out', 'Auto', tou_dyn_filter),
            'notoptout_manual': filter_df(df_2022_all, 'Not opt out', 'Manual', tou_dyn_filter),
            'non_intervention': df_2022_all[df_2022_all['Anlagenart'].notnull()& df_2022_all['Category'].isnull() & df_2022_all['Control'].isnull()]
        },
        '2023': {
            'optin': filter_df(df_2023_all, 'Optin', tou_dyn_filter=tou_dyn_filter),
            'notoptout': filter_df(df_2023_all, 'Not opt out', tou_dyn_filter=tou_dyn_filter),
            'optin_auto': filter_df(df_2023_all, 'Optin', 'Auto', tou_dyn_filter),
            'optin_manual': filter_df(df_2023_all, 'Optin', 'Manual', tou_dyn_filter),
            'notoptout_auto': filter_df(df_2023_all, 'Not opt out', 'Auto', tou_dyn_filter),
            'notoptout_manual': filter_df(df_2023_all, 'Not opt out', 'Manual', tou_dyn_filter),
            'non_intervention': df_2023_all[df_2023_all['Anlagenart'].notnull() & df_2023_all['Category'].isnull() & df_2023_all['Control'].isnull()]
        },
        '2024': {
            'optin': filter_df(df_2024_all, 'Optin', tou_dyn_filter=tou_dyn_filter),
            'notoptout': filter_df(df_2024_all, 'Not opt out', tou_dyn_filter=tou_dyn_filter),
            'optin_auto': filter_df(df_2024_all, 'Optin', 'Auto', tou_dyn_filter),
            'optin_manual': filter_df(df_2024_all, 'Optin', 'Manual', tou_dyn_filter),
            'notoptout_auto': filter_df(df_2024_all, 'Not opt out', 'Auto', tou_dyn_filter),
            'notoptout_manual': filter_df(df_2024_all, 'Not opt out', 'Manual', tou_dyn_filter),
            'non_intervention': df_2024_all[df_2024_all['Anlagenart'].notnull() & df_2024_all['Category'].isnull() & df_2024_all['Control'].isnull()]
        }
    }
    year_colors = {
        '2022': '#6ead58',  # Color for the year 2022
        '2023': '#77c8ff',  # Color for the year 2023
        '2024': '#ffa66b'  # Color for the year 2024
    }

    num_rows = len(building_types) + int(include_others) + int(include_total)

    fig, axes = plt.subplots(nrows=num_rows, ncols=len(plot_columns), figsize=(7 * len(plot_columns), 3 * num_rows),constrained_layout=True)


    if num_rows == 1 and len(plot_columns) == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or len(plot_columns) == 1:
        axes = np.atleast_2d(axes)

    for row, building_type in enumerate(building_types):
        for col, title in enumerate(plot_columns):
            ax = axes[row][col] if num_rows > 1 or len(plot_columns) > 1 else axes[0, 0]
            plot_data_exists = False
            for year, df_dict in dataframes_per_year.items():
                if nom_year == 2023 and year == '2022':
                    continue
                df = df_dict[title.lower().replace(" ", "")]
                filtered_df = df[df['Anlagenart'] == building_type]

                if not filtered_df.empty:
                    plot_data_exists = True
                    mean_activeP_per_slot = filtered_df.groupby(['slot', 'Anlagenart'])['ActiveP'].mean().reset_index()
                    if normalised and year in ['2023', '2024']:
                        mean_activeP_per_slot = normalize(df_norm, mean_activeP_per_slot, year, building_type, nom_year)
                        mean_activeP_per_slot = mean_activeP_per_slot.reset_index(drop=True)

                    mean_activeP_per_slot['time'] = mean_activeP_per_slot['slot'].apply(slot_to_time)
                    color = year_colors.get(year)
                    ax.plot(mean_activeP_per_slot['time'], mean_activeP_per_slot['ActiveP'], label=f'{year}',color=color,linewidth=3)

            if plot_data_exists:
                english_name = building_types_map.get(building_type, building_type)
                num_buildings = filtered_df['aim_geraet_ldn'].nunique()
                ax.set_title(f'{title.replace("_", " ")} for {english_name} ({num_buildings} buildings)',fontsize=14)
                #ax.set_xlabel('Time Period(Hrs)',fontsize=15)
                if col == 0:
                    ax.set_ylabel('Mean Active Power (kW)', fontsize=14)
                ax.set_xticks([])
                ax.set_xticklabels([])
                #ax.set_xticks(mean_activeP_per_slot['time'][::6])
                #ax.set_xticklabels(mean_activeP_per_slot['time'][::6], rotation=45,fontsize=15)
                ax.set_autoscale_on(True)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
                ax.legend()
                if tou_dyn_filter == 'Tou' and period == 'Summer':
                    ax.axvspan(72, 96, color='red', alpha=0.2)
                    ax.axvspan(28, 52, color='red', alpha=0.2)
                    ax.axvspan(52, 72, color='green', alpha=0.2)
                    ax.axvspan(0, 28, color='#fce5cd', alpha=0.2)

                elif tou_dyn_filter == 'Tou':
                    ax.axvspan(72, 96, color='red', alpha=0.2)
                    ax.axvspan(0, 72, color='green', alpha=0.2)
                elif tou_dyn_filter == 'Dyn':
                    df = peak_ident(start_date, end_date)
                    peak_slots = sorted(df[df['period_type'] == 'peak']['slot'].tolist())
                    off_peak_slots = sorted(df[df['period_type'] == 'off-peak']['slot'].tolist())

                    def group_consecutive_slots(slots):
                        ranges = []
                        start = slots[0]
                        end = slots[0]

                        for i in range(1, len(slots)):
                            if slots[i] == end + 1:
                                end = slots[i]
                            else:
                                ranges.append((start, end + 0.5))
                                start = slots[i]
                                end = slots[i]
                        ranges.append((start, end + 0.5))
                        return ranges

                    color_slots = {
                        'red': group_consecutive_slots(peak_slots),
                        'green': group_consecutive_slots(off_peak_slots)
                    }

                    for color, ranges in color_slots.items():
                        for start, end in ranges:
                            ax.axvspan(start, end, color=color, alpha=0.2)
            else:
                ax.axis('off')

    if include_others:
        others_row = len(building_types)
        for col, title in enumerate(plot_columns):
            ax = axes[others_row][col]
            plot_data_exists = False
            for year, df_dict in dataframes_per_year.items():
                if nom_year == 2023 and year == '2022':
                    continue
                df = df_dict[title.lower().replace(" ", "")]
                filtered_df = df[~df['Anlagenart'].isin(building_types)]

                if not filtered_df.empty:
                    plot_data_exists = True
                    mean_activeP_per_slot = filtered_df.groupby(['slot'])['ActiveP'].mean().reset_index()
                    if normalised and year in ['2023', '2024']:
                        mean_activeP_per_slot = normalize(df_norm, mean_activeP_per_slot, year, building_type=None,
                                                          nom_year=nom_year)
                        mean_activeP_per_slot = mean_activeP_per_slot.reset_index(drop=True)
                    mean_activeP_per_slot['time'] = mean_activeP_per_slot['slot'].apply(slot_to_time)
                    color = year_colors.get(year)
                    ax.plot(mean_activeP_per_slot['time'], mean_activeP_per_slot['ActiveP'], label=f'{year}',color=color,linewidth=3)

            if plot_data_exists:
                english_name = building_types_map.get(building_type, building_type)

                num_buildings = filtered_df['aim_geraet_ldn'].nunique()
                ax.set_title(f'{title.replace("_", " ")} for Others ({num_buildings} buildings)',fontsize=14)
                #ax.set_xlabel('Time Period(Hrs)',fontsize=15)
                if col == 0:
                    ax.set_ylabel('Mean Active Power (kW)', fontsize=14)
                ax.set_xticks([])
                ax.set_xticklabels([])
                #ax.set_xticks(mean_activeP_per_slot['time'][::6])
                #ax.set_xticklabels(mean_activeP_per_slot['time'][::6], rotation=45,fontsize=15)
                ax.set_autoscale_on(True)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
                ax.legend()

                if tou_dyn_filter == 'Tou' and period == 'Summer':
                    ax.axvspan(72, 96, color='red', alpha=0.2)
                    ax.axvspan(28, 52, color='red', alpha=0.2)
                    ax.axvspan(52, 72, color='green', alpha=0.2)
                    ax.axvspan(0, 28, color='#fce5cd', alpha=0.2)

                elif tou_dyn_filter == 'Tou':
                    ax.axvspan(72, 96, color='red', alpha=0.2)
                    ax.axvspan(0, 72, color='green', alpha=0.2)

                elif tou_dyn_filter == 'Dyn':
                    df = peak_ident(start_date, end_date)
                    peak_slots = sorted(df[df['period_type'] == 'peak']['slot'].tolist())
                    off_peak_slots = sorted(df[df['period_type'] == 'off-peak']['slot'].tolist())

                    def group_consecutive_slots(slots):
                        ranges = []
                        start = slots[0]
                        end = slots[0]

                        for i in range(1, len(slots)):
                            if slots[i] == end + 1:
                                end = slots[i]
                            else:
                                ranges.append((start, end + 0.5))
                                start = slots[i]
                                end = slots[i]
                        ranges.append((start, end + 0.5))
                        return ranges

                    color_slots = {
                        'red': group_consecutive_slots(peak_slots),
                        'green': group_consecutive_slots(off_peak_slots)
                    }

                    for color, ranges in color_slots.items():
                        for start, end in ranges:
                            ax.axvspan(start, end, color=color, alpha=0.2)
            else:
                ax.axis('off')

    if include_total:
        total_row = len(building_types) + int(include_others)
        for col, title in enumerate(plot_columns):
            ax = axes[total_row][col]
            plot_data_exists = False
            for year, df_dict in dataframes_per_year.items():
                if nom_year == 2023 and year == '2022':
                    continue
                df = df_dict[title.lower().replace(" ", "")]
                mean_activeP_per_slot = df.groupby(['slot'])['ActiveP'].mean().reset_index()

                if not mean_activeP_per_slot.empty:
                    plot_data_exists = True
                    if normalised and year in ['2023', '2024']:
                        mean_activeP_per_slot = normalize(df_norm, mean_activeP_per_slot, year, building_type='all', nom_year=nom_year)
                        mean_activeP_per_slot = mean_activeP_per_slot.reset_index(drop=True)
                    mean_activeP_per_slot['time'] = mean_activeP_per_slot['slot'].apply(slot_to_time)
                    color = year_colors.get(year)
                    ax.plot(mean_activeP_per_slot['time'], mean_activeP_per_slot['ActiveP'], label=f'{year}',color=color,linewidth=3)

            if plot_data_exists:
                num_buildings = df['aim_geraet_ldn'].nunique()
                ax.set_title(f'{title.replace("_", " ")} Total ({num_buildings} buildings)', fontsize= 14)
                ax.set_xlabel('Time Period(Hrs)',fontsize=14)
                if col == 0:
                    ax.set_ylabel('Mean Active Power (kW)', fontsize=14)
                ax.set_xticks(mean_activeP_per_slot['time'][::6])
                ax.set_xticklabels(mean_activeP_per_slot['time'][::6], rotation=45,fontsize=15)
                ax.set_autoscale_on(True)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
                ax.legend()

                if tou_dyn_filter == 'Tou' and period == 'Summer':
                    ax.axvspan(72, 96, color='red', alpha=0.2)
                    ax.axvspan(28, 52, color='red', alpha=0.2)
                    ax.axvspan(52, 72, color='green', alpha=0.2)
                    ax.axvspan(0, 28, color='#fce5cd', alpha=0.2)

                elif tou_dyn_filter == 'Tou':
                    ax.axvspan(72, 96, color='red', alpha=0.2)
                    ax.axvspan(0, 72, color='green', alpha=0.2)
                elif tou_dyn_filter == 'Dyn':
                    df = peak_ident(start_date, end_date)
                    peak_slots = sorted(df[df['period_type'] == 'peak']['slot'].tolist())
                    off_peak_slots = sorted(df[df['period_type'] == 'off-peak']['slot'].tolist())

                    def group_consecutive_slots(slots):
                        ranges = []
                        start = slots[0]
                        end = slots[0]

                        for i in range(1, len(slots)):
                            if slots[i] == end + 1:
                                end = slots[i]
                            else:
                                ranges.append((start, end + 0.5))
                                start = slots[i]
                                end = slots[i]
                        ranges.append((start, end + 0.5))
                        return ranges

                    color_slots = {
                        'red': group_consecutive_slots(peak_slots),
                        'green': group_consecutive_slots(off_peak_slots)
                    }

                    for color, ranges in color_slots.items():
                        for start, end in ranges:
                            ax.axvspan(start, end, color=color, alpha=0.2)
            else:
                ax.axis('off')
    plt.savefig(f'All_{"normalised" if normalised else "non_normalised"}_{tou_dyn_filter}_{nom_year}_{period}.pdf')
    plt.savefig(f'All_{"normalised" if normalised else "non_normalised"}_{tou_dyn_filter}_{nom_year}_{period}.jpeg')
    plt.show()

# Function to plot tariff/charges comparison for customer category for period:
def plot_single_period_distributions(df_anlys, category,remove_outliers=True, period='winter'):
    # Define the date ranges for each period
    periods = {
        'summer': ('2024-05-01', '2024-05-31'),
        'winter': ('2023-12-01', '2024-02-29'),
        'transition': ('2024-03-01', '2024-04-30')
    }
    df_anlys = df_anlys[df_anlys['Category'].str.contains(category, na=False)]
    # Validate the period
    if period not in periods:
        raise ValueError(f"Invalid period '{period}'. Choose from {list(periods.keys())}.")

    # Define colors for regular, OrtsNetz, and differences
    colors = {
        'Network_cost': 'skyblue',
        'OrtsNetz_Network_cost': 'lightgreen',
        'Energy_cost': 'salmon',
        'OrtsNetz_Energy_cost': 'lightcoral',
        'Diff_Network_cost': 'lightblue',
        'Diff_Energy_cost': 'peachpuff'
    }

    # Filter data by date range (assuming 'Date' is a valid column in `df_anlys`)
    #start_date, end_date = periods[period]
    #df_period = df_anlys[(df_anlys['Date'] >= start_date) & (df_anlys['Date'] <= end_date)]

    # Exclude cases with zero values in relevant columns
    df_period = df_anlys[(df_anlys['Network_cost'] != 0) &
                         (df_anlys['OrtsNetz_Network_cost'] != 0) &
                         (df_anlys['Energy_cost'] != 0) &
                         (df_anlys['OrtsNetz_Energy_cost'] != 0)]

    # Aggregate the data by 'Geschäftspartner' for average costs
    agg_avg = {
        'Network_cost': 'mean',
        'OrtsNetz_Network_cost': 'mean',
        'Energy_cost': 'mean',
        'OrtsNetz_Energy_cost': 'mean'
    }
    df_total = df_period.groupby('Geschäftspartner').agg(agg_avg).reset_index()

    # Calculate differences
    df_total['Diff_Network_cost'] = df_total['OrtsNetz_Network_cost'] - df_total['Network_cost']
    df_total['Diff_Energy_cost'] = df_total['OrtsNetz_Energy_cost'] - df_total['Energy_cost']

    # Remove outliers if specified
    if remove_outliers:
        def remove_iqr_outliers(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

        for col in agg_avg.keys():
            df_total = remove_iqr_outliers(df_total, col)
        df_total = remove_iqr_outliers(df_total, 'Diff_Network_cost')
        df_total = remove_iqr_outliers(df_total, 'Diff_Energy_cost')

    # Set up 1x6 plot layout
    fig, axs = plt.subplots(1, 6, figsize=(24, 6))  # Adjust figsize for better readability

    # Define the columns and labels for the 1x6 layout
    plot_info = [
        ('Network_cost', 'Network Cost (CHF)'),
        ('OrtsNetz_Network_cost', 'OrtsNetz Network Cost (CHF)'),
        ('Energy_cost', 'Energy Cost (CHF)'),
        ('OrtsNetz_Energy_cost', 'OrtsNetz Energy Cost (CHF)'),
        ('Diff_Network_cost', 'Actual Saving Network Cost (CHF)'),
        ('Diff_Energy_cost', 'Actual Saving Energy Cost (CHF)')
    ]

    # Plot each distribution for the current period with specific colors
    for ax, (col, ylabel) in zip(axs, plot_info):
        sns.violinplot(data=df_total, y=col, ax=ax, color=colors[col])
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True)

    # Perform Mann-Whitney U tests between regular and OrtsNetz costs
    tests = [
        ('Network_cost', 'OrtsNetz_Network_cost', 'Network Cost'),
        ('Energy_cost', 'OrtsNetz_Energy_cost', 'Energy Cost')
    ]

    for regular, ortsnetz, label in tests:
        # Extract the data for each category
        data_regular = df_total[regular]
        data_ortsnetz = df_total[ortsnetz]

        # Perform the Mann-Whitney U test
        stat, p_value = mannwhitneyu(data_regular, data_ortsnetz, alternative='two-sided')

        # Print the test results
        print(f"Mann-Whitney U Test for {label}:")
        print(f"U statistic = {stat}, p-value = {p_value}\n")

        # Interpretation
        if p_value < 0.05:
            print(f"There is a significant difference in {label} between Regular and OrtsNetz (p < 0.05).\n")
        else:
            print(f"No significant difference in {label} between Regular and OrtsNetz (p >= 0.05).\n")

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(category, y=1.02, fontsize=20)  # Add a super title with some margin at the top
    plt.savefig(f'{category}_cost_dist_violin_with_differences.pdf')
    plt.savefig(f'{category}_cost_dist_violin_with_differences.jpeg')
    plt.show()

def plot_all_period_distributions_cost(df_anlys, remove_outliers=True):
    # Define the date ranges for each period
    periods = {
        'Winter': ('2023-12-01', '2024-02-29'),
        'Transition': ('2024-03-01', '2024-04-30'),
        'Summer': ('2024-05-01', '2024-05-31')
    }

    # Define colors for each plot
    colors = {
        'Network_saving': 'skyblue',
        'Energy_saving': 'lightgreen',
        'Diff_Network_cost': 'lightblue',
        'Diff_Energy_cost': 'peachpuff'
    }

    # Set up 3x4 plot layout
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))  # Adjust figsize for better readability

    # Iterate over the periods and corresponding subplot row
    for period_index, (period, (start_date, end_date)) in enumerate(periods.items()):
        # Filter data by date range
        df_period = df_anlys[(df_anlys['Date'] >= start_date) & (df_anlys['Date'] <= end_date)]

        # Exclude cases with zero values in relevant columns
        df_period = df_period[(df_period['Network_cost'] != 0) &
                              (df_period['OrtsNetz_Network_cost'] != 0) &
                              (df_period['Energy_cost'] != 0) &
                              (df_period['OrtsNetz_Energy_cost'] != 0)]

        # Aggregate the data by 'Geschäftspartner' for average costs
        agg_avg = {
            'Network_cost': 'mean',
            'OrtsNetz_Network_cost': 'mean',
            'Energy_cost': 'mean',
            'OrtsNetz_Energy_cost': 'mean',
            'Energy_saving': 'mean',
            'Network_saving': 'mean'
        }
        df_total = df_period.groupby('Geschäftspartner').agg(agg_avg).reset_index()

        # Calculate differences
        df_total['Diff_Network_cost'] = df_total['OrtsNetz_Network_cost'] - df_total['Network_cost']
        df_total['Diff_Energy_cost'] = df_total['OrtsNetz_Energy_cost'] - df_total['Energy_cost']

        # Remove outliers if specified
        if remove_outliers:
            def remove_iqr_outliers(data, column):
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

            for col in agg_avg.keys():
                df_total = remove_iqr_outliers(df_total, col)
            df_total = remove_iqr_outliers(df_total, 'Diff_Network_cost')
            df_total = remove_iqr_outliers(df_total, 'Diff_Energy_cost')

        # Define the columns and labels for the row corresponding to the period
        plot_info = [
            ('Network_saving', 'Network Savings via best accounting(CHF)'),
            ('Energy_saving', 'Energy Savings via best accounting(CHF)'),
            ('Diff_Network_cost', 'Actual Network Cost Savings(CHF)'),
            ('Diff_Energy_cost', 'Actual Energy Cost Savings(CHF)')
        ]

        # Plot each distribution for this period in its corresponding row
        for column_index, (col, ylabel) in enumerate(plot_info):
            ax = axs[period_index, column_index]
            sns.violinplot(data=df_total, y=col, ax=ax, color=colors[col])

            ax.set_title(f'{period.capitalize()}-{ylabel}', fontsize=14)

            ax.set_xlabel('')
            ax.set_ylabel(ylabel, fontsize=14)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('all_period_cost_distribution_plots.pdf')
    plt.savefig('all_period_cost_distribution_plots.jpeg')
    plt.show()


if __name__ == "__main__":
    # Enrollment data from Consumer_energy
    df_all,df_main, df_BP_of_interest, filtered_df_int, df_optin,df_optin_dyn,df_optin_tou, df_notoptout,df_notoptout_dyn,df_notoptout_tou,filtered_optin, filtered_not_opt_out, df_auto_wout_dir,filtered_auto_wout_dir,df_manu, filtered_manu, filtered_optin_auto,filtered_optin_manu,filtered_not_optout_auto,filtered_not_optout_manu = process()
    dfs = [df_main,filtered_df_int,filtered_optin, filtered_not_opt_out]

    # Input is energy and cost sheet for 2022 year generated from Consumer_energy file
    # Distribution all Cost and Energy
    #distribution_all(dfs, labels, col='Total_energy', x_min=-5000, x_max=20000)
    #distribution_all(dfs, labels,col='Total_cost', x_min=-50000, x_max=20000)
    '''
    # Title: All,Optin,notoptout
    #Individual Distributions (Energy)
    #distribution(filtered_optin, title='Optin', col='Total_energy')
    #distribution(filtered_df_int, col='Total_energy')
    #distribution(filtered_not_opt_out, title='notoptout', col='Total_energy')
    #distribution(df_main,title='All', col='Total_energy')

    #Individual Distributions (Cost)
    #distribution(filtered_optin, title='Optin', col='Total_cost')
    #distribution(filtered_df_int, col='Total_cost')
    #distribution(filtered_not_opt_out, title='notoptout', col='Total_cost')
    #distribution(df_main,title='All', col='Total_cost')

    # Average Cost and Average Energy by building type
    #plot_building_metric(dfs,labels,col='Total_cost')
    #plot_building_metric(dfs ,labels,col='Total_energy')


    #Percentage of different building types in different consumer categories
    #plot_percent_builtype(dfs,labels)

    # Percentage Energy Ratio multiple categories
    #plot_energy_ratio(dfs, labels)
    #plot_energy_ratio_distribution_multiple(dfs, labels )


    # Percentage Energy Ratio Individual categories
    #plot_energy_ratio_distribution(filtered_optin,category='Opt in consumers')
    #plot_energy_ratio_distribution(df_main,category='All consumers')
    #plot_energy_ratio_distribution(filtered_df_int,category='Consumers of Interest')
    #plot_energy_ratio_distribution(filtered_not_opt_out,category='Notoptout consumers')


    # Individual Box Plots
    #box_plot_low_energy_pct(df_main, title='ALL consumers')
    #box_plot_low_energy_pct(filtered_df_int, title='Consumers of Interest')
    #box_plot_low_energy_pct(filtered_optin, title='Opt in consumers')
    #box_plot_low_energy_pct(filtered_not_opt_out, title='Not Opt Out consumers')


    # Combined Box plots
    #box_plot_multiple_dfs(dfs, labels, filter=True)
    #box_plot_multiple_dfs(dfs, labels, filter=False)

    #Vilon Plots
    # Low Energy percentage Filtered and Unfiltered
    #violin_plot_multiple_dfs(dfs,labels, title="Percentage Energy during Low Energy Tariff for different consumer categories", col ='low_energy_pct',filter=True)
    #violin_plot_multiple_dfs(dfs,labels, title="Percentage Energy during Low Energy Tariff for different consumer categories", col ='low_energy_pct',filter=True)

    # Cost and Energy
    #violin_plot_multiple_dfs(dfs,labels, title="Total energy consumed during low tariff period for different consumer categories", col ='Total_energy',filter=True)
    #violin_plot_multiple_dfs(dfs,labels, title="Total Cost during low tariff period for different consumer categories", col ='Total_cost',filter=True)

    #plot_building_total(dfs, labels, 'Total_cost')
    #plot_building_total(dfs, labels, 'Total_energy')

    # Input is Daily Averaged Load profile files
    # Process the load data files and plot curves
    columns_to_plot = ['Non_intervention', 'Optin_auto','Notoptout_auto','Optin_manual', 'Notoptout_manual']
    # Function to read the Excel files and sheets
    df_2022_build, df_2022_mean, df_2022_all,df_2023_build, df_2023_mean, df_2023_all, df_2024_build, df_2024_mean, df_2024_all = readfiles()
    # Function to plot daily profiles
    #1.Input plot_columns = ['Non_intervention', 'Optin_auto','Notoptout_auto','Optin_manual', 'Notoptout_manual'] any combination
    #2.normalised = True/False
    #3.nom_year = 2022/2023 whether to normalise based on 2023,2024 and for naming
    #4.tou_dyn_filter = 'Tou'/'Dyn'/None
    #5.Period= 'Winter'/'Summer'/'Transition' accordingly peak periods for TOU changes
    process_plot_loadprofile(df_2022_all, df_2023_all, df_2024_all, building_types, plot_columns=columns_to_plot, normalised=True, nom_year=2022, tou_dyn_filter='Dyn',include_others=True, include_total=True,period='Summer')

    # Plots for Tariff comparison:
    # Input is generated tariff comparison sheet from Consumer_energy.py
    df_tarif= pd.read_excel('Tariffs_comparison_mapped.xlsx')

    # Comparison of network cost, energy cost and actual savings in network cost and energy cost for a period and category
    plot_single_period_distributions(df_tarif, category='Not opt out', remove_outliers=True, period='summer')
    plot_single_period_distributions(df_tarif, category='Not opt out', remove_outliers=True, period='transition')
    plot_single_period_distributions(df_tarif, category='Not opt out', remove_outliers=True, period='winter')

    # Comparison of network cost, energy cost and actual savings in network cost and energy cost for a period and category
    plot_single_period_distributions(df_tarif, category='Opt in', remove_outliers=True, period='summer')
    plot_single_period_distributions(df_tarif, category='Opt in', remove_outliers=True, period='transition')
    plot_single_period_distributions(df_tarif, category='Opt in', remove_outliers=True, period='winter')

    # To plot Comparison of actual savings and savings as per best accounting policy for different periods
    plot_all_period_distributions_cost(df_tarif,remove_outliers=True)
    '''

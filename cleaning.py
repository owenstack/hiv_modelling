import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set Seaborn style
sns.set_style("whitegrid")

# 1. Load Data and Initial Inspection
print("\n1. LOADING AND INITIAL INSPECTION")
print("-" * 50)

# Read the Excel file with correct columns
df = pd.read_excel('data/data.xlsx')

# Keep only relevant columns
df = df[['date', 'enrollment']]
df['date'] = pd.to_datetime(df['date'])

print("First few rows of the data:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())
print("\nInitial Statistics:")
print(df.describe())

# Check for duplicate dates before reindexing
duplicates = df[df['date'].duplicated()].copy()
if not duplicates.empty:
    print("\nDuplicate dates found:")
    print(duplicates.sort_values('date'))
    # Aggregate duplicates by summing enrollments
    df = df.groupby('date')['enrollment'].sum().reset_index()

# 2. Address Missing Dates
print("\n2. ADDRESSING MISSING DATES")
print("-" * 50)

# Create complete date range
full_date_range = pd.date_range(start='2007-01-01', end='2023-12-31', freq='D')
df.set_index('date', inplace=True)

# Reindex with the complete date range
df_complete = df.reindex(full_date_range)
print(f"Number of NaN values after reindexing: {df_complete.isna().sum()}")

# 3. Investigate Zero Enrollments
print("\n3. ZERO ENROLLMENTS ANALYSIS")
print("-" * 50)

# Create day of week mapping
dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
           4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

df_complete = df_complete.reset_index()
df_complete['DayOfWeek'] = df_complete['index'].dt.dayofweek.map(dow_map)
df_complete['Month'] = df_complete['index'].dt.strftime('%B')
df_complete['Year'] = df_complete['index'].dt.year
df_complete['IsWeekend'] = df_complete['index'].dt.dayofweek >= 5

# Enhanced zero enrollments analysis
zero_enrollments = df_complete[df_complete['enrollment'] == 0]
total_zeros = len(zero_enrollments)
total_days = len(df_complete)
zero_percentage = (total_zeros / total_days) * 100

print(f"\nZero Enrollments Summary:")
print(f"Total zero enrollment days: {total_zeros}")
print(f"Percentage of zero enrollment days: {zero_percentage:.2f}%")

# Analyze zeros by day of week
zero_by_day = df_complete[df_complete['enrollment'] == 0].groupby('DayOfWeek').size()
total_by_day = df_complete.groupby('DayOfWeek').size()
mean_by_day = df_complete.groupby('DayOfWeek')['enrollment'].mean()

print("\nZero enrollments analysis by day of week:")
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day in days_of_week:
    total = total_by_day.get(day, 0)
    zeros = zero_by_day.get(day, 0)
    zero_pct = (zeros / total * 100) if total > 0 else 0
    print(f"{day}:")
    print(f"  Total days: {total}")
    print(f"  Zero enrollment days: {zeros}")
    print(f"  Percentage zeros: {zero_pct:.2f}%")
    print(f"  Mean enrollments: {mean_by_day.get(day, 0):.2f}")

# Visualize zero enrollment patterns
plt.figure(figsize=(12, 6))
zero_by_day.plot(kind='bar')
plt.title('Zero Enrollments by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Zero Enrollment Days')
plt.tight_layout()
plt.savefig('plots/zero_enrollments_by_day.png')
plt.close()

# Monthly trend analysis
monthly_stats = df_complete.groupby(['Year', 'Month']).agg({
    'enrollment': ['mean', 'std', 'count', 
                  lambda x: (x == 0).sum(),
                  lambda x: ((x == 0).sum() / len(x)) * 100]
}).reset_index()
monthly_stats.columns = ['Year', 'Month', 'Mean', 'Std', 'Count', 'Zeros', 'Zero_Percentage']

# Yearly trend analysis
yearly_stats = df_complete.groupby('Year').agg({
    'enrollment': ['mean', 'std', 'count', 
                  lambda x: (x == 0).sum(),
                  lambda x: ((x == 0).sum() / len(x)) * 100]
}).reset_index()
yearly_stats.columns = ['Year', 'Mean', 'Std', 'Count', 'Zeros', 'Zero_Percentage']

print("\nYearly Trends Summary:")
print(yearly_stats.to_string(index=False))

# Visualize yearly trends with numpy arrays
years = np.array(yearly_stats['Year'])
means = np.array(yearly_stats['Mean'])
stds = np.array(yearly_stats['Std'])

plt.figure(figsize=(15, 5))
plt.plot(years, means, marker='o', label='Mean Enrollments')
plt.fill_between(years, 
                means - stds,
                means + stds,
                alpha=0.2,
                label='±1 Standard Deviation')
plt.title('Yearly Enrollment Trends with Standard Deviation')
plt.xlabel('Year')
plt.ylabel('Mean Daily Enrollments')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plots/yearly_trends.png')
plt.close()

# Additional trend visualization
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_complete, x='Year', y='enrollment')
plt.title('Enrollment Distribution by Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/yearly_distributions.png')
plt.close()

# Monthly seasonality visualization
plt.figure(figsize=(15, 6))
monthly_means = df_complete.groupby('Month')['enrollment'].mean()
monthly_means = monthly_means.reindex(['January', 'February', 'March', 'April', 'May', 'June',
                                     'July', 'August', 'September', 'October', 'November', 'December'])
plt.bar(range(len(monthly_means)), monthly_means, tick_label=monthly_means.index)
plt.title('Average Enrollments by Month')
plt.xlabel('Month')
plt.ylabel('Mean Daily Enrollments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/monthly_trends.png')
plt.close()

# 4. Check for Duplicate Dates
print("\n4. CHECKING FOR DUPLICATE DATES")
print("-" * 50)
print(f"Number of duplicate dates: {df_complete['index'].duplicated().sum()}")

# 5. Examine Outliers
print("\n5. OUTLIER ANALYSIS")
print("-" * 50)

# Enhanced outlier analysis
Q1 = df_complete['enrollment'].quantile(0.25)
Q3 = df_complete['enrollment'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
extreme_outlier_threshold = Q3 + 3 * IQR

# Find extreme outliers
extreme_outliers = df_complete[df_complete['enrollment'] > extreme_outlier_threshold]
print(f"\nExtreme outliers (>{extreme_outlier_threshold:.2f}):")
if not extreme_outliers.empty:
    # Get top 5 extreme outliers
    print("\nTop 5 extreme outliers:")
    top_5_mask = extreme_outliers['enrollment'].nlargest(5).index
    top_5_outliers = extreme_outliers.loc[top_5_mask]
    for _, row in top_5_outliers.iterrows():
        print(f"Date: {row['index'].strftime('%Y-%m-%d')}, Enrollments: {int(row['enrollment'])}")
else:
    print("No extreme outliers found")

# Create plots
plt.figure(figsize=(15, 5))
plt.boxplot(df_complete['enrollment'].dropna())
plt.title('Box Plot of Daily Enrollments')
plt.ylabel('Number of Enrollments')
plt.savefig('plots/enrollments_boxplot.png')
plt.close()

plt.figure(figsize=(15, 5))
plt.plot(df_complete['index'], df_complete['enrollment'])
plt.title('Time Series of Daily Enrollments')
plt.xlabel('Date')
plt.ylabel('Number of Enrollments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/enrollments_timeseries.png')
plt.close()

# 6. Verify Data Validity
print("\n6. DATA VALIDITY CHECK")
print("-" * 50)
print(f"Number of negative values: {(df_complete['enrollment'] < 0).sum()}")
non_integer_mask = df_complete['enrollment'].notna() & (df_complete['enrollment'] % 1 != 0)
print(f"Number of non-integer values: {non_integer_mask.sum()}")

# 7. Handle Missing Values
print("\n7. HANDLING MISSING VALUES")
print("-" * 50)
df_complete['enrollment_cleaned'] = df_complete['enrollment'].fillna(0)
print(f"Original NaN count: {df_complete['enrollment'].isna().sum()}")
print(f"Cleaned NaN count: {df_complete['enrollment_cleaned'].isna().sum()}")

# 8. Final Review
print("\n8. FINAL REVIEW")
print("-" * 50)
print("\nDescriptive statistics of cleaned data:")
print(df_complete['enrollment_cleaned'].describe())

# Create monthly averages
monthly_avg = df_complete.groupby([df_complete['Year'], 
                                 df_complete['Month']])['enrollment_cleaned'].mean()
print("\nMonthly averages (first 5 months):")
print(monthly_avg.head())

# Plot cleaned time series
plt.figure(figsize=(15, 5))
plt.plot(df_complete['index'], df_complete['enrollment_cleaned'])
plt.title('Time Series of Cleaned Daily Enrollments')
plt.xlabel('Date')
plt.ylabel('Number of Enrollments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/enrollments_cleaned_timeseries.png')
plt.close()

# Statistical summary
print("\nDetailed Statistical Summary:")
print("-" * 50)
print("\nYearly Statistics:")
print(yearly_stats.to_string(index=False))

print("\nMonthly Statistics (across all years):")
monthly_summary = df_complete.groupby('Month')['enrollment'].agg(['mean', 'std', 'count']).round(2)
print(monthly_summary.to_string())

# Save cleaned data
df_complete.to_csv('data/cleaned_enrollments.csv', index=False)
print("\nAnalysis complete. Cleaned data and visualizations have been saved.")
# HIV Patient Enrollment Data Analysis

This project analyzes daily HIV patient enrollment data from 2007 to 2023, performing comprehensive data cleaning, validation, and statistical analysis.

## Dataset Overview

- **Time Period**: 2007-01-01 to 2023-12-31
- **Original Data**: Excel file (`data.xlsx`) with daily enrollment counts
- **Key Statistics**:
  - Mean: 17.56 enrollments/day
  - Median: 12 enrollments/day
  - Standard Deviation: 24.84
  - Range: 0-198 enrollments/day

## Analysis Steps

### 1. Data Loading and Initial Inspection
- Loaded data from Excel file
- Converted dates to datetime format
- Performed initial statistical analysis

### 2. Missing Dates Analysis
- Identified gap between expected dates (6209) and actual records (6024)
- Created complete daily date range
- Reindexed data to include all dates
- Missing dates were later filled with zeros based on domain knowledge

### 3. Zero Enrollment Analysis
Key findings were saved in `zero_enrollments_by_day.png`, revealing:
- Distribution of zero-enrollment days across weekdays
- Weekend vs weekday enrollment patterns
- Monthly and yearly zero-enrollment trends

### 4. Duplicate Date Check
- Identified and handled any duplicate date entries
- Aggregated duplicate entries by summing enrollments

### 5. Outlier Analysis
Generated visualizations:
- `enrollments_boxplot.png`: Distribution and outliers
- `enrollments_timeseries.png`: Full time series view
- Identified extreme outliers (>Q3 + 3*IQR)

### 6. Data Quality Verification
- Checked for negative values
- Verified integer consistency
- Validated date continuity

### 7. Time Series Analysis
Generated multiple visualizations:
- `yearly_trends.png`: Annual enrollment patterns with confidence bands
- `yearly_distributions.png`: Year-by-year distribution comparison
- `monthly_trends.png`: Monthly seasonality patterns

### 8. Final Cleaned Dataset
- Saved as `cleaned_enrollments.csv`
- Includes all dates from 2007-2023
- Missing values filled with zeros
- Original and derived features included

## Visualizations

The analysis produced several visualizations:
1. `zero_enrollments_by_day.png`: Zero enrollment patterns by day of week
2. `yearly_trends.png`: Year-over-year enrollment trends with standard deviation bands
3. `yearly_distributions.png`: Box plots showing enrollment distributions by year
4. `monthly_trends.png`: Monthly enrollment patterns
5. `enrollments_boxplot.png`: Distribution and outlier visualization
6. `enrollments_timeseries.png`: Complete time series visualization
7. `enrollments_cleaned_timeseries.png`: Cleaned data time series

## Data Files

- `data.xlsx`: Original data file
- `cleaned_enrollments.csv`: Processed and cleaned dataset

## Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- seaborn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Analysis

Execute the analysis script:
```bash
python model.py
```

This will generate all visualizations and the cleaned dataset, along with detailed statistical output in the console.

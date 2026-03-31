import pandas as pd
import numpy as np

# --- Setup: Create dummy 'hr_data.csv' based on the handwritten output table ---
dummy_csv_data = {
    'Name': ['raj', 'suresh', 'Suresh', 'dinesh', 'mahesh', 'ramya', 'ramya'],
    'Salary': [20000.0, 30000.0, 30000.0, 30000.0, np.nan, 100000.0, np.nan],
    'ReviewDate': ['20-02-25', '30-03-25', '30-03-25', '04-03-25', np.nan, '4-4-24', '4-4-24'],
    'Performance Score': [9.8, 5.6, 5.6, 4.4, 8.0, 5.8, 8.8]
}
pd.DataFrame(dummy_csv_data).to_csv('hr_data.csv', index=False)
# -------------------------------------------------------------------------------

# Load the dataset
df = pd.read_csv('hr_data.csv')
print("Original DataFrame:\n", df)

# 1. Remove duplicate entries
df = df.drop_duplicates()

# 2. Handle missing values (Fill missing salaries with the column mean)
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# 3. Transform column formats (Convert to datetime)
df['ReviewDate'] = pd.to_datetime(df['ReviewDate'], format='mixed', dayfirst=True)

# 4. Ensure performance score is numeric
df['Performance Score'] = pd.to_numeric(df['Performance Score'])

# Save the cleaned dataset to a new CSV
df.to_csv('cleaned_hr_data.csv', index=False)

print("\nCleaned DataFrame:\n", df)

# Data Analysis
print("\nThe maximum score is:", df['Performance Score'].max())
print("The minimum score is:", df['Performance Score'].min())
import pandas as pd

print("Test case 1")
# covid_df = pd.read_csv('covid_19.csv')

# Dummy data for testing
data = {
    'District': ['Bengaluru', 'Mysuru', 'DK', 'Udupi', 'Belagavi'],
    'Cases': [100, 450, 300, 150, 200],
    'Deaths': [15, 4, 3, 1, 2],
    'Recovery_Rate': [95.5, 96.2, 94.8, 97.1, 98.0] # Added one value to match array lengths
}

covid_df = pd.DataFrame(data)
print(covid_df.head(2)) # First 2 rows

print("\nTest case 2")
emp_data = {'Age': [25, 30, 45], 'Salary': [50000, 60000, 9000]}
employee_df = pd.DataFrame(emp_data)
print(employee_df.describe())

print("\nData structure info:")
covid_df.info()
print("\n")

print("Operations:")
avg = covid_df['Cases'].mean()
print(f"Avg of cases: {avg}")

var_recovery = covid_df['Recovery_Rate'].var()
print(f"Recovery Rate variance: {var_recovery:.2f}")

std_deaths = covid_df['Deaths'].std()
print(f"Std of deaths: {std_deaths:.2f}")

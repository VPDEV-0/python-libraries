import numpy as np

# --- Test Case 1: Weekly Rainfall Data ---
print("Rainfall Data")
rainfall_data = np.array([12.5, 10.0, 45.2, 8.5])
rainfall_std = np.std(rainfall_data)
print(f"Std of Rainfall : {rainfall_std:.2f} mm\n")

# --- Test Case 2: Student Exam Scores ---
print("Test Case 2")
exam_scores = np.array([85, 92, 78, 90])
score_mean = np.mean(exam_scores)
score_median = np.median(exam_scores)
print(f"Mean (Average) Score: {score_mean:.2f}")
print(f"Median Score : {score_median:.2f}\n")

# --- Scenario: Climate Stability ---
print("Climate Stability")
daily_temps = np.array([30.5, 31.0, 29.8, 30.2, 30.5, 29.5, 40.0])

temp_mean = np.mean(daily_temps)
temp_variance = np.var(daily_temps)
temp_std = np.std(daily_temps)

print(f"Avg temp : {temp_mean:.2f} C")
print(f"Temp variance : {temp_variance:.2f}")
print(f"Std : {temp_std:.2f} C")

historical_stability = 2.5
if temp_std > historical_stability:
    print("\nConclusion : Std is high")
else:
    print("\nConclusion : Temp is stable")
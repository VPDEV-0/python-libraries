


import numpy as np

def run_manufacturing_scenario():
    print("--- 1. Manufacturing Quality Control ---")
    # A factory measures gear diameters. Extreme outliers must be capped at 5.0mm to prevent errors.
    deviation = np.array([[1.2, 5.8, 2.1],
                          [8.9, 1.5, 6.2],
                          [3.0, 4.1, 1.1]])
    
    average_deviation = np.mean(deviation)
    capped_deviations = np.clip(deviation, 0, 5.0)
    flat_log = capped_deviations.ravel()
    
    print(f"Overall Average Deviation: {average_deviation:.2f} mm")
    print(f"\nCapped Deviations Matrix:\n{capped_deviations}")
    print(f"\nDatabase Log Format:\n{flat_log}\n")


def run_climate_scenario():
    print("--- 2. Climate Data Processing ---")
    # A meteorologist analyzes temp data over 3 days. Day 3 (column index 2) had a calibration error, needs +2 degrees.
    temp = np.array([[22, 24, 21],
                     [25, 26, 23],
                     [19, 20, 18]])
    
    # Add 2 to all temperatures recorded on that specific day (3rd column)
    temp[:, 2] = temp[:, 2] + 2
    
    # Flatten into single row format for an upcoming ML model (1 row, 9 columns)
    flat_temp = temp.reshape(1, 9)
    
    print(f"Corrected Temp Matrix:\n{temp}")
    print(f"\nFlattened Format:\n{flat_temp}\n")


def run_hr_scenario():
    print("--- 3. HR Performance Ratings ---")
    # 3 employees across 3 projects. Ratings < 3 get a +1 boost before reporting.
    ratings = np.array([[4, 2, 3],
                        [3, 5, 2],
                        [2, 4, 5]])
    
    adjusted = np.where(ratings < 3, ratings + 1, ratings)
    report = adjusted.flatten()
    
    print(f"Adjusted Ratings:\n{adjusted}")
    print(f"\nPerformance Report:\n{report}\n")


def run_assessment_scenario():
    print("--- 4. Assessment Weighting ---")
    # Each assessment has different weightage (A1=1, A2=2, A3=1.5). Apply weights automatically.
    marks = np.array([[10, 20, 15],
                      [22, 19, 12],
                      [25, 23, 21]])
    
    weights = np.array([1, 2, 1.5])
    
    # Multiply marks by weights using broadcasting
    weighted_marks = marks * weights
    
    # Reshape to a single row
    report = weighted_marks.reshape(1, -1)
    
    print(f"Weighted marks:\n{weighted_marks}")
    print(f"\nSingle-row report:\n{report}\n")


if __name__ == "__main__":
    run_manufacturing_scenario()
    run_climate_scenario()
    run_hr_scenario()
    run_assessment_scenario()


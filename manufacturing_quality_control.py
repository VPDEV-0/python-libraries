import numpy as np

# A factory manufactures metal gears. 
# Sensor measures diameter of gears produced by 3 different production batches.
deviation = np.array([[1.2, 5.8, 2.1],
                      [8.9, 1.5, 6.2],
                      [3.0, 4.1, 1.1]])

average_deviation = np.mean(deviation)

# Cap extreme outliers at 5.0mm
capped_deviations = np.clip(deviation, 0, 5.0)

# Log into a flat database record
flat_log = capped_deviations.ravel()

print(f"Overall Average Deviation : {average_deviation:.2f} mm")
print("\nCapped Deviations matrix :\n", capped_deviations)
print("\nDatabase log Format :\n", flat_log)
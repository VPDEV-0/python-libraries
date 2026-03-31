import numpy as np

# 3 different weather stations over 3 days
temp = np.array([[22, 24, 21],
                 [25, 26, 23],
                 [19, 20, 18]])

# Add 2 degrees to the temperatures recorded on Day 3 (Index 2)
temp[:, 2] = temp[:, 2] + 2

# Flatten into a single row format
flat_temp = temp.reshape(1, 9)

print("Corrected temp matrix :\n", temp)
print("Flattened Format :\n", flat_temp)
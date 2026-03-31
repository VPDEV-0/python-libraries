import numpy as np

marks = np.array([[10, 20, 15],
                  [22, 19, 17],
                  [25, 23, 21]])

# Each assessment has different weightage (A1=1, A2=2, A3=1.5)
weight = np.array([1, 2, 1.5])

weighted_marks = marks * weight
report = weighted_marks.reshape(1, -1)

print("Weighted marks :\n", weighted_marks)
print("Single-row report :\n", report)
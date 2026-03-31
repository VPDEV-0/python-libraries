import numpy as np

# Performance ratings of 3 employees across 3 projects
ratings = np.array([[4, 2, 3],
                    [3, 5, 2],
                    [2, 4, 5]])

# Low rating (<3) gets a performance boost before reporting
adjusted = np.where(ratings < 3, ratings + 1, ratings)

report = adjusted.flatten()

print("Adjusted Ratings :\n", adjusted)
print("Performance Report :\n", report)
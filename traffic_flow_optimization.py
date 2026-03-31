import numpy as np

# Traffic matrix representing vehicles going between Point 1 and Point 2
A = np.array([[200, 150],
              [100, 250]])

# Transformation matrix (e.g., 80% current flow maintained, 20% rerouted)
T = np.array([[0.8, 0.2],
              [0.3, 0.7]])

# Simulates new traffic after signal optimization
A_new = np.dot(A, T)

print("Original traffic matrix:")
print(A)
print("Transformation matrix:")
print(T)
print("New traffic matrix after signal optimization:")
print(A_new)
print("Total traffic before optimization:", np.sum(A))
print("Total traffic after optimization:", np.sum(A_new))

print("\n--- Break Statement Snippet ---")
# Stops when number equals 5
for i in range(10):
    if i == 5:
        break
    print(i)
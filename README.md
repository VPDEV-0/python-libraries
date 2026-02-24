# NumPy Practical Data Processing Scenarios

This repository contains practical examples of using Python's NumPy library to solve real-world data manipulation and array processing problems. 

## Scenarios Covered

1. **Manufacturing Quality Control**
   - **Goal:** Cap extreme outliers in gear diameter measurements before logging them into a database.
   - **Key Concepts:** `np.array`, `np.mean()`, `np.clip()`, flattening arrays with `.ravel()`.

2. **Climate Data Processing**
   - **Goal:** Correct a specific day's calibration error in a temperature matrix and flatten the data for a machine learning model.
   - **Key Concepts:** Array slicing/indexing (updating a specific column), broadcasting, and reshaping with `.reshape()`.

3. **HR Performance Ratings**
   - **Goal:** Identify low employee performance ratings across projects and apply a +1 bonus score before reporting.
   - **Key Concepts:** Conditional array logic using `np.where()`, flattening arrays with `.flatten()`.

4. **Assessment Weighting**
   - **Goal:** Apply different weightages to multiple student assessment scores automatically.
   - **Key Concepts:** Vector multiplication, array broadcasting, and reshaping to a single-row format using `.reshape(1, -1)`.

## Prerequisites
To run this code, you need Python installed along with the NumPy library.
```bash
pip install numpy

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

# Python Data Processing & Analysis

This repository contains a collection of Python scripts demonstrating real-world data processing, statistical analysis, and data cleaning scenarios using `numpy` and `pandas`. 

These scripts were created as practice exercises to handle common data manipulation tasks.

## 📂 Repository Contents

### NumPy Operations & Statistics
* **`manufacturing_quality_control.py`**: Uses NumPy to cap extreme outliers in production data and flatten matrices for database logging.
* **`climate_data_processing.py`**: Demonstrates array indexing and matrix reshaping to fix sensor calibration errors in climate data.
* **`hr_performance_ratings.py`**: Uses `np.where` to conditionally adjust employee performance ratings and flattens the report.
* **`weight_bonus_calculation.py`**: Applies mathematical operations on NumPy arrays to calculate weighted assessment marks.
* **`traffic_flow_optimization.py`**: Uses matrix dot products (`np.dot`) to simulate and evaluate traffic redistribution across intersections.
* **`statistical_functions.py`**: Calculates mean, median, variance, and standard deviation for real-world datasets (exam scores, rainfall, temperature stability).

### Pandas Data Exploration & Cleaning
* **`pandas_data_exploration.py`**: Covers the basics of loading DataFrames, viewing data structure (`info()`, `describe()`, `head()`), and computing column-specific statistics.
* **`hr_data_cleaning.py`**: A complete data cleaning pipeline that removes duplicate records, imputes missing values (`fillna`) using column averages, and standardizes date formats.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries. You can install them using pip:

```bash
pip install numpy pandas

import pandas as pd

# Load the cars dataset
cars = pd.read_csv('C:\\Users\\Abhishek\\Desktop\\6_SEM\\PS-2\\pr1\\USA_cars_datasets.csv')

# Display basic information about the dataset
print("1] Summary Statistics for cars dataset:\n")
print(cars.describe())

# Display the structure of the dataset
print("\n2] Structure Information for cars dataset:\n")
print(cars.info())

# Display quartiles of the numerical columns
print("\n3] Quartile Information for cars dataset:\n")
print(cars['mileage'].quantile([0.25, 0.5, 0.75]))

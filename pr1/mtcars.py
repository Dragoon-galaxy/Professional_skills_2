import pandas as pd

mtcars = pd.read_csv('C:\\Users\\Abhishek\\Desktop\\6_SEM\\PS-2\\pr1\\mtcars.csv')

#  Display summary statistics
print("1] Summary Statistics for mtcars dataset:\n")
print(mtcars.describe())

#  Display structure information
print("\n2] Structure Information for mtcars dataset:\n")
print(mtcars.info())

#  Use the quantile() method to get the quartile values for a specific column
print("\n3] Quartile Information for mtcars dataset:\n")
print(mtcars['mpg'].quantile([0.25, 0.5, 0.75]))

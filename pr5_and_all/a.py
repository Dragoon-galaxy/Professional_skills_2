# Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load the dataset
url = "https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/admission.csv"

dataset = pd.read_csv(url)

# Rename columns if necessary for better readability
dataset.columns = ['admitted', 'gre_score', 'gpa', 'rank']

# Define the independent and dependent variables
X = dataset[['gre_score', 'gpa', 'rank']]
y = dataset['admitted']

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit logistic regression model
model = sm.Logit(y, X).fit()

# Print summary of the model
print(model.summary())

# Fit the null model
null_model = sm.Logit(y, np.ones((X.shape[0], 1))).fit()

# Calculate residual deviance
resid_deviance = -2 * (null_model.llf - model.llf)

# Calculate p-value of chi-squared test
p_value = stats.chi2.sf(resid_deviance, model.df_resid)

# Print p-value of chi-squared test
print('\np-value:', p_value)

# Check if the model is fit or not based on the p-value
if p_value > 0.05:
    print("Model is fit.")
else:
    print("Model is unfit.")

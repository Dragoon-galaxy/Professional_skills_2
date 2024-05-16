import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Compute the correlation matrix
correlation_matrix = iris_df.corr()

# Print the correlation matrix
print("CORRELATION MATRIX")
print(correlation_matrix)

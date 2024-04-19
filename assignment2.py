import pandas as pd
from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Here we're going to transform the variables 
#---------------------------------------------
# Load the dataset
data = pd.read_csv('INSERT HERE YOUR PATH WHERE YOU HAVE THE .csv file')
data # I'm checking if dataset has been loaded correctly

# Clean the DataFrame by removing the row with transformation codes
data_cleaned = data.drop(index=0)
data_cleaned.reset_index(drop=True, inplace=True)
data_cleaned['sasdate'] = pd.to_datetime(data_cleaned['sasdate'], format='%m/%d/%Y')
data_cleaned

# Extract transformation codes
transformation_codes = data.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# 'transformation_codes' has the variable’s name (Series) and its transformation 
# (Transformation_Code). There are six possible transformations
#to see them-->
transformation_codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$

# Function to apply transformations based on the transformation code
#   1) We're defining 'apply_transformation' function

def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

#   2) Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    data_cleaned[series_name] = apply_transformation(data_cleaned[series_name].astype(float), float(code))


data_cleaned = data_cleaned[2:]     # We drop the first two observations of the dataset
data_cleaned.reset_index(drop=True, inplace=True) #We reset the index, now  first 
                                                # observation has index 0
data_cleaned.head()

# The data are ready to be used
#------------------------------


# LET'S DEVELOP OUR AR(7) MODEL

# Dependent variable INDPRO
Y = data_cleaned['INDPRO']
Y

# The AR(7) likelihood
#--------------------------
# DEFINING FUNCTIONS 
import numpy as np
import scipy 
def unconditional_ar_mean_variance(c, phis, sigma2): # 'c' = the constant term of the AR(7) model, 
                                                     # 'phis' = vector containing autoregressive coefficients of order 1 to 7
                                                     # 'sigma2' = the variance of the process.
    ## The length of phis is p
    p = len(phis)
    A = np.zeros((p, p))                             # 'A' = tridiagonal matrix in which the phis coefficients are on the 
                                                     # first row and the main diagonal contains 1.
    A[0, :] = phis
    A[1:, 0:(p-1)] = np.eye(p-1)
    ## Check for stationarity: checking whether all absolute values of the eigenvalues of A are less than 1. 
    # If this condition is met, the process is stationary and the stationary variable is set to True, otherwise to False.
    eigA = np.linalg.eig(A)
    if all(np.abs(eigA.eigenvalues)<1):
        stationary = True
    else:
        stationary = False
    # Create the vector b, a vector of zeros
    b = np.zeros((p, 1))
    b[0, 0] = c
    
    # Compute the mean using matrix algebra (Stationary mean)
    I = np.eye(p)
    mu = np.linalg.inv(I - A) @ b 
    
    # Solve the discrete Lyapunov equation
    Q = np.zeros((p, p))
    Q[0, 0] = sigma2
    #Sigma = np.linalg.solve(I - np.kron(A, A), Q.flatten()).reshape(7, 7)
    Sigma = scipy.linalg.solve_discrete_lyapunov(A, Q)
    
    return mu.ravel(), Sigma, stationary


from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np
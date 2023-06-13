import pandas as pd
import statsmodels.api as sm


x1=[328.58, 324.99, 323.94, 331.65, 335.33, 335.22, 334.25, 325.93, 332.29]
x2=[181.27, 181.5, 177.9, 178.44, 179.97, 182.63, 181.03, 177.7, 177.33]
y=[430.92, 429.96, 426.62, 428.44, 426.67, 428.28, 424.5, 418.09, 418.28]

# Create a DataFrame with your time series data
data = pd.DataFrame({
    'X1':x1,
    'X2':x2,
    'Y': y
})

# Separate the independent variables (X1, X2) and the dependent variable (Y)
X = data[['X1', 'X2']]
Y = data['Y']

# Add a constant column to the independent variables (for intercept)
X = sm.add_constant(X)

# Fit the multiple regression model
model = sm.OLS(Y, X)
results = model.fit()

# Get the coefficients
coefficients = results.params

# Print the coefficients
print(coefficients)

# Perform a t-test for a specific coefficient
hypothesis = "X1 = 0"
t_test = results.t_test(hypothesis)
print("# t_test:", t_test)
#==============================

# Mean of x1, x2 and y
mean_x1 = sum(x1) / len(x1)
mean_x2 = sum(x2) / len(x2)
mean_y = sum(y) / len(y)

# Variance of x1 and x2
var_x1 = sum((i - mean_x1)**2 for i in x1) / len(x1)
var_x2 = sum((i - mean_x2)**2 for i in x2) / len(x2)

# Covariance of x1,y and x2,y
covar_x1_y = sum((x1[i] - mean_x1) * (y[i] - mean_y) for i in range(len(x1))) / len(x1)
covar_x2_y = sum((x2[i] - mean_x2) * (y[i] - mean_y) for i in range(len(x2))) / len(x2)

# Coefficients
b1 = covar_x1_y / var_x1
b2 = covar_x2_y / var_x2

# Intercept
alpha = mean_y - b1*mean_x1 - b2*mean_x2

print(b1, b2, alpha)
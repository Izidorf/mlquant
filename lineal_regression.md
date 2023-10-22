## 1. What is linear regression and what are the model assumptions?

Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables. It is a foundational method in statistics and machine learning used for predicting numerical outcomes.

### Simple Linear Regression:

In simple linear regression, there's one independent variable and one dependent variable. The relationship is represented as:
 $`\ \mathbf{ y = \beta_0 + \beta_1x + \epsilon } `$ where elements of $`\ \mathbf{\epsilon} `$ are in $`\ \mathbf{ \mathcal{N} (0, \sigma^2) } `$


Where:
- $`\ y `$ is the dependent variable.
- $`\ x `$ is the independent variable.
- $`\ \beta_0 `$ is the intercept.
- $`\ \beta_1 `$ is the slope of the line.
- $`\ \epsilon `$ represents the error term (difference between observed and predicted values).

### Multiple Linear Regression:

For multiple independent variables, the relationship becomes:  

$`\ \mathbf{ y  = \beta_0 + \beta_1x_1 + \beta_2x_2 + } ... + \beta_nx_n +  \epsilon  `$ where elements of $`\ \mathbf{\epsilon} `$ are in $`\ \mathbf{ \mathcal{N} (0, \sigma^2) } `$

or in vector notation:  

 $`\  \mathbf{Y = X \beta + \epsilon}  `$  where elements of $`\ \mathbf{\epsilon} `$ are in $`\ \mathbf{ \mathcal{N} (0, \sigma^2) } `$


### Assumptions of Linear Regression:

1. **Linearity**: The relationship between the independent and dependent variables is linear. This can often be checked using scatter plots.
2. **Independence**: The observations $`\ x_1,x_2...x_n `$  are independent of each other. This is more of a study design issue than something you can test.
3. **Homoscedasticity of Residuals (Equal Variance)**: The variance of the residuals (errors, $`\ \epsilon `$) is constant across all levels of the independent variables. This can be checked using a residual vs. fitted values plot.
4. **Normality of Residuals**: The residuals (or errors, $`\ \epsilon `$) should be normally distributed. This can be checked using histograms, Q-Q plots, or statistical tests like the Shapiro-Wilk test.
5. **No Autocorrelation of Residuals**: In time series data, residuals should not have autocorrelation. This can be tested using the Durbin-Watson test.
6. **No Multicollinearity**: In multiple linear regression, the independent variables should not be too highly correlated with each other. This can be checked using variance inflation factors (VIF), correlation matrices, or condition indices.
7. **Exogenity/No Endogeneity**: The error term $`\ \epsilon `$ should not be correlated with the independent variables.

If these assumptions are violated, the reliability and interpretability of the regression coefficients may be compromised. Diagnostic tests and plots can check for these assumptions, and there are remedies to address violations.

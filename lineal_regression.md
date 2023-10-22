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


## 2. What happens if two features in linear regression are correlated and how to deal with this situation?

When two or more features (or independent variables) in a linear regression model are correlated, it leads to a phenomenon called **collinearity or multicollinearity**. Here's what happens and its implications:

1. **Variance of the Coefficient Estimates:** The variance of the coefficient estimates can increase significantly. This means that the estimates of the coefficients can become very sensitive to small changes in the model. Thus, the model becomes unstable. 
In the presence of perfect multicollinearity, the ordinary least squares (OLS) estimates for the regression coefficients are not identifiable; that is, there are infinitely many combinations of the coefficients that could fit the data perfectly. As a result, the sampling distribution of the estimator (beta-hat) will have infinite variance.   &nbsp;<br /> 
To understand this more intuitively, let's break it down:

    1. **Matrix Inversion Problem:** The OLS estimator for the regression coefficients in matrix notation is given by: $`\ β^=(X′X)−1X′Yβ^​=(X′X)−1X′Y `$ where $`\ X′X′ `$ is the transpose of the matrix $`\ XX `$ (which contains the predictor values) and YY is the response variable vector. Perfect multicollinearity means that the matrix $`\ X′XX′X `$ is singular (i.e., it has a determinant of zero). Consequently, the inverse, $`\ (X′X)−1(X′X)−1 `$, doesn't exist. Thus, the OLS estimator cannot be computed as defined.

    2. **Geometric Interpretation:** Consider a simple regression with two perfectly correlated predictors. In a 2-dimensional space, these predictors fall on a straight line. When we try to project the dependent variable values onto the space spanned by the predictors (i.e., fit the model), there's an infinite number of ways to split the influence between the two predictors because they provide the same information. This leads to infinite possible values for the regression coefficients, and therefore, infinite variance for their estimates.

    3. **Intuitive Explanation:** Suppose you're trying to predict a person's weight based on two predictors: their height in inches and their height in centimeters. These two predictors are perfectly correlated (since one is just a scaled version of the other). Now, there's no unique way to determine the individual contribution of each predictor to the weight because any increase in height in inches corresponds to an increase in height in centimeters. The model can't distinguish between their effects, leading to infinite possibilities for the coefficients.

While perfect multicollinearity results in infinite variance for the OLS estimates, it's worth noting that in many practical scenarios, you'll encounter near multicollinearity (where predictors are highly but not perfectly correlated). In such cases, the variance of the OLS estimates won't be infinite, but it can be very large, making the estimates unstable and less reliable.


2. **Coefficients Interpretation:** The interpretation of the coefficients becomes challenging. A significant coefficient in the presence of multicollinearity might not actually imply a strong relationship between the predictor and the response variable, because the effect of that predictor might be confounded with the effect of another correlated predictor.
3. **Significance Tests:** Because of the increased variance of the coefficient estimates, significance tests for the coefficients can be unreliable. A predictor might be statistically significant in one sample but not another. Or, in the presence of multicollinearity, predictors that are actually influential might be deemed not statistically significant.
4. **Prediction:** While multicollinearity affects the interpretation and reliability of the coefficients, it doesn't necessarily degrade the predictive capability of the model as a whole. If the primary goal is prediction (rather than interpretation of coefficients), multicollinearity might be less of a concern.
5. **Condition Number:** The condition number, which measures the sensitivity of the output to changes in the input, can become large in the presence of multicollinearity. A large condition number indicates numerical instability in the calculations, which can affect the accuracy of the estimated coefficients.
6. **Eigenvalues and Variance Inflation Factor (VIF):** When checking for multicollinearity, small eigenvalues and large VIFs indicate the presence of multicollinearity. The VIF measures how much the variance of a coefficient is increased due to multicollinearity. A common rule of thumb is that a VIF above 5-10 indicates high multicollinearity.

### Remedies for Multicollinearity:

1. **Remove Variables:** Remove one of the correlated variables.
   2. **Domain Knowledge Variable Removal:** This might be done based on domain knowledge, where you choose to keep the variable that has a clearer interpretation or is more relevant to the study.
   1. **Regularization as Variable Selection:** Techniques like Ridge Regression or Lasso introduce a penalty term for the size of coefficients, which can help in handling multicollinearity.

3. **Combine Variables:** Sometimes, correlated variables can be combined to form a single variable. For instance, using an average or principal component analysis (PCA) to reduce dimensionality.

4. **Increase Sample Size:** Sometimes, increasing the sample size can help mitigate the effects of multicollinearity.

5. **Centering the Variables:** Subtracting the mean from each observation can sometimes help, though this doesn't eliminate multicollinearity, it can make it easier to deal with.

In summary, multicollinearity can be a concern in linear regression, especially when the goal is inference about individual predictors. It's essential to check for multicollinearity when building a regression model and to take appropriate measures if it's present.

## 3. What is $`\ R^2 `$?

$`\ \mathbf{R^2} `$ (pronounced "R-squared") is the coefficient of determination. It's a statistical measure used in the context of regression analysis to assess how well the regression model fits the observed data. Specifically, $`\ \mathbf{R^2} `$ represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

### Calculation:

In the context of a simple linear regression:

1. **SST** (Total Sum of Squares) represents the total variance in the dependent variable, and is calculated as:
$`\ SST = \sum\limits_{i=1}^{n} (y_i - \bar{y})^2 `$
where $`\ y_i `$ are the observed values, $`\ \bar{y} `$ is the mean of the observed values, and $`\ n `$ is the number of observations.

2. **SSR** (Sum of Squares due to Regression) represents the amount of variance explained by the regression model, and is calculated as:
$`\ SSR = \sum\limits_{i=1}^{n} (\hat{y}_i - \bar{y})^2 `$
where $`\ \hat{y}_i `$ are the predicted values from the regression model.

3. **SSE** (Sum of Squares of Errors) represents the variance that's not explained by the model, and is calculated as:
$`\ SSE = \sum\limits_{i=1}^{n} (y_i - \hat{y}_i)^2 `$

Given the above, $`\ \mathbf{R^2} `$ is defined as:
$`\ R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST} `$

### Interpretation:

- An $`\ \mathbf{R^2} `$ value of 1 indicates that the regression model perfectly explains the variability of the dependent variable around its mean. In other words, all of the variance in the dependent variable is captured by the model.
  
- An $`\ \mathbf{R^2} `$ value of 0 indicates that the model does not explain any of the variability of the dependent variable around its mean.

- Generally, a higher $`\ \mathbf{R^2} `$ suggests a better fit of the model to the data. However, a high $`\ \mathbf{R^2} `$ does not necessarily mean the model is appropriate. For example, if a model is overfitted, it might have a high $`\ \mathbf{R^2} `$ on the training data but perform poorly on new, unseen data.

### Caveats:

1. **Doesn’t Imply Causation**: A high $`\ \mathbf{R^2} `$ value doesn’t imply causation between the independent and dependent variables.

2. **Comparing Models**: While $`\ \mathbf{R^2} `$ can be useful for comparing different models on the same dataset, it may not be ideal for comparing models across different datasets.

3. **Adjusted $`\ \mathbf{R^2} `$**: This is a modification of $`\ \mathbf{R^2} `$ that adjusts for the number of predictors in a model. As more predictors are added to a model, $`\ \mathbf{R^2} `$ will generally increase (even if those predictors are not truly meaningful). Adjusted $`\ \mathbf{R^2} `$ takes into account the complexity of the model and can decrease if irrelevant predictors are added.

4. **Limitations in Nonlinear Models**: $`\ \mathbf{R^2} `$ is most interpretable in the context of linear regression. In nonlinear models, the interpretation of $`\ \mathbf{R^2} `$ can be less straightforward.

In summary, $`\ \mathbf{R^2} `$ provides a measure of how well the observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model. However, like all statistical metrics, it should be interpreted with caution and in the context of the specific analysis and dataset.




## 4. What is is L1 and L2 Regularization?

L1 and L2 regularization are techniques used to prevent overfitting in machine learning models, especially in linear regression. They work by adding penalty terms to the original loss function, constraining the magnitude of the coefficients in the model. 

### L1 Regularization (Lasso):

**Formulation:**  
$`\ \text{Loss} = \text{Original Loss} + \lambda \sum\limits_{i=1}^{p} | \beta_i | `$

Where:
- $`\ \beta_i `$ are the model coefficients.
- $`\ \lambda `$ is a positive regularization parameter. Higher values of $`\ \lambda `$ result in stronger regularization.
- $`\ p `$ is the number of features in the model.

**Characteristics:**
- Produces sparse solutions. Some coefficients can become exactly zero, effectively leading to feature selection.
- Often used when we have a large number of features and we believe many of them are irrelevant or redundant.
- Because of its property of setting coefficients to zero, L1 can be less stable in the presence of highly correlated features; it might select one feature and ignore the others.

### L2 Regularization (Ridge):

**Formulation:**  
$`\ \text{Loss} = \text{Original Loss} + \lambda \sum\limits_{i=1}^{p} \beta_i^2 `$

Where:
- $`\ \beta_i `$ are the model coefficients.
- $`\ \lambda `$ is a positive regularization parameter. Higher values of $`\ \lambda `$ result in stronger regularization.
- $`\ p `$ is the number of features in the model.

**Characteristics:**
- Does not produce sparse solutions. Coefficients are shrunk toward zero, but they rarely become exactly zero.
- Tends to handle multicollinearity better than L1 by distributing the coefficient magnitude among correlated features.
- Generally more stable than L1 when features are correlated.

### Elastic Net:


Elastic Net is a combination of L1 and L2 regularization. It adds both L1 and L2 penalty terms to the loss function:

$`\ \text{Loss} = \text{Original Loss} + \lambda_1 \sum\limits_{i=1}^{p} | \beta_i | + \lambda_2 \sum\limits_{i=1}^{p} \beta_i^2 `$

Where $`\ \lambda_1 `$ and $`\ \lambda_2 `$ control the strength of L1 and L2 penalties, respectively.

**Characteristics:**
- Combines the properties of L1 and L2 regularization.
- Can produce sparse solutions (due to the L1 term) while also handling multicollinearity (due to the L2 term).



## 6. Write a closed-form solution for linear regression with L2 regularisation, L1 regularisation and without regularisation? 

## 7.
## 8.
## 9.
## 10.

## . How would you estimate the error in linear regression from the beta?



## 5. What is the difference between regressing Y on X vs X on Y? How do the two $`\ R^2 `$ relate to each other?

In order to answer this question it is important to organise thought process and come up with a basic set of assumptions, which could make answering that question easier. For example: 
- make assumptions
- write down two models
- derive betas
- look at the problem from geometric perspective
- analyse $`\ R^2 `$

Assumptions:
$`\ E[x]=E[y]=0 `$

Model 1

$$ \eqalign{ y&=\beta_1 x + \varepsilon_1, \quad \varepsilon_1 \sim \mathbb{N}(0,\sigma_1 I_n)  \\
O L S: \hat{\beta}_1 &=\frac{x^T y}{x^{\top} x} \approx \frac{Cov(x, y)}{\mathbb{V}(x)} } $$

Model 2

$$ \eqalign{ x&=\beta_2 y + \varepsilon_2, \quad \varepsilon_2 \sim \mathbb{N}(0,\sigma_2 I_n)  \\
O L S: \hat{\beta}_1 &=\frac{y^T x}{y^{\top} y} \approx \frac{Cov(x, y)}{\mathbb{V}(y)} } $$

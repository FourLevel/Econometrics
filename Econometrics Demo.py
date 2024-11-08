# Import Libraries
import numpy as np # For numerical operations
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For plotting
import statsmodels.api as sm # For statistical modeling
from scipy import stats # For statistical tests
from statsmodels.stats.outliers_influence import variance_inflation_factor # For checking multicollinearity
from statsmodels.stats.diagnostic import het_breuschpagan # For checking heteroscedasticity
from statsmodels.stats.stattools import durbin_watson # For checking autocorrelation
from linearmodels.panel import PanelOLS # For performing regression analysis


# Read Excel file
file_path = '銀行資料_丟迴歸分析_20240523.csv'
data = pd.read_csv(file_path)

# Select variables
df = data[['公司代碼', '年', 'Z-score', '資產總額', '淨收益', '當季季底P/B', 
           '負債比率', '公司年齡', '董事總持股數%', '經理人總持股%', 
           '是否在金融危機時期', '是否在COVID時期']]
df.columns = ['company_code', 'year', 'z_score', 'total_assets', 'net_income', 'pb_ratio', 
              'debt_ratio', 'company_age', 'shareholders_percentage', 'managers_percentage', 
              'crisis_period', 'covid_period']
df = df.set_index(['company_code', 'year'])

# Check missing values
print(df.isnull().sum())

# Standardize quantitative variables and add them to df
# Define columns to standardize
columns_to_standardize = [
    'total_assets', 'net_income', 'pb_ratio', 'debt_ratio',
    'company_age', 'shareholders_percentage', 'managers_percentage']

# Create standardized column names
standardized_columns = ['std_' + col for col in columns_to_standardize]

# Perform standardization
df[standardized_columns] = df[columns_to_standardize].apply(
    lambda x: (x - x.mean()) / x.std())

# Set y and X
y = df['z_score']
X_quantitative = df[['std_total_assets', 'std_net_income', 'std_pb_ratio', 'std_debt_ratio', 
                     'std_company_age', 'std_shareholders_percentage', 'std_managers_percentage']]

# Check Correlation Matrix
plt.figure(figsize=(10, 6), dpi=200)
plt.title('Correlation Matrix')
sns.heatmap(X_quantitative.corr(), annot=True, cmap='coolwarm')
plt.show()


''' Single factor analysis: Check Variable Significance '''
# Perform OLS regression analysis for each X_quantitative variable against y, and organize p-values into a list
pvalue_results = pd.DataFrame(columns=['Variable', 'p-value'])

for column in X_quantitative.columns:
    # Use current variable for OLS regression analysis
    formula = f'y ~ {column} + EntityEffects'
    fixed_effects_model = PanelOLS.from_formula(formula, data=df)
    fixed_effects_results = fixed_effects_model.fit()
    
    # Get p-value of current variable
    p_value = fixed_effects_results.pvalues[column]
    new_row = pd.DataFrame({'Variable': [column], 'p-value': [f"{p_value:.4f}"]})
    pvalue_results = pd.concat([pvalue_results, new_row], ignore_index=True)

print(pvalue_results)


''' Choose variables and perform Regression Model '''
# Choose X variables
X_model_1 = df[['std_total_assets', 'std_pb_ratio', 'std_debt_ratio', 
                'std_company_age', 'std_managers_percentage', 'crisis_period']]

# Regression Model
fixed_effects_model = PanelOLS.from_formula('y ~ std_total_assets + std_pb_ratio + std_debt_ratio + std_company_age + std_managers_percentage + crisis_period + EntityEffects', data=df)
fixed_effects_results = fixed_effects_model.fit()

# Display regression model summary
print(fixed_effects_results.summary)


''' Check Linear Relationship '''
# Plot dependent variable and independent variables
plt.figure(figsize=(10, 10), dpi=200)
plt.suptitle('Scatter Plot of Independent Variables vs Dependent Variable')
for i, col in enumerate(X_model_1.columns):
    plt.subplot(3, 3, i + 1)
    plt.scatter(X_model_1[col], y, alpha=1)
    plt.title(f'{col} vs Z-score')
    plt.xlabel(col)
    plt.ylabel('Z-score')
plt.tight_layout()
plt.show()


''' Check Multi-collinearity '''
# Check Correlation Matrix
plt.figure(figsize=(10, 10), dpi=200)
plt.title('Correlation Matrix')
sns.heatmap(X_model_1.corr(), annot=True, cmap='coolwarm')
plt.show()

# Check Variance Inflation Factor (VIF)
X_model_1 = sm.add_constant(X_model_1)
# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_model_1.columns
vif_data["VIF"] = [variance_inflation_factor(X_model_1.values, i) for i in range(X_model_1.shape[1])]

# Display results
print("VIF results：")
print(vif_data)

# VIF explanation
print("\nVIF explanation：")
print("VIF < 5：No serious multicollinearity")
print("5 <= VIF < 10：Moderate multicollinearity")
print("VIF >= 10：Serious multicollinearity, suggest handling")

# Identify variables that need to be handled
problematic_vars = vif_data[vif_data["VIF"] >= 10]["Variable"].tolist()
if problematic_vars:
    print(f"\nVariables need to be handled：{', '.join(problematic_vars)}")
else:
    print("\nAll variables' VIF values are within acceptable range")


''' Check Autocorrelation '''
# Check Durbin-Watson Test
# Fit regression model
model = sm.OLS(y, X_model_1).fit()

# Calculate Durbin-Watson statistic
dw_statistic = durbin_watson(model.resid)
print("\nDurbin-Watson Test results：")
print(f"Durbin-Watson statistic：{dw_statistic:.4f}")

# Durbin-Watson statistic explanation
print("\nDurbin-Watson statistic explanation：")
print("0 to 2：Positive autocorrelation")
print("2：No autocorrelation")
print("2 to 4：Negative autocorrelation")

if dw_statistic < 1.5:
    print("\nConclusion：Significant positive autocorrelation")
elif dw_statistic > 2.5:
    print("\nConclusion：Significant negative autocorrelation")
else:
    print("\nConclusion：No significant autocorrelation")


''' Check Heteroscedasticity '''
# Fit regression model
model = sm.OLS(y, X_model_1).fit()

# Perform Breusch-Pagan test
bp_test = het_breuschpagan(model.resid, model.model.exog)

labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
bp_result = dict(zip(labels, bp_test))

# Print Breusch-Pagan test results
print("\nBreusch-Pagan Test:")
for key, value in bp_result.items():
    print(f"  {key}: {value}")

# Short interpretation of results
print("\nShort interpretation of results：")
print("If the p-value is less than 0.05, the null hypothesis can be rejected, indicating the presence of heteroscedasticity.")
print("If the p-value is greater than 0.05, the null hypothesis cannot be rejected, indicating no heteroscedasticity.")

'''
Interpretation of results：
Lagrange multiplier statistic：This is the test statistic for the Breusch-Pagan test. It measures the degree of heteroscedasticity in the model. The value itself does not have a direct interpretation, but is used in conjunction with the p-value.
p-value：This is the p-value associated with the test statistic. It represents the probability of observing a test statistic as extreme as the one observed, assuming the null hypothesis (no heteroscedasticity) is true. Typically, a significance level of 0.05 is used. If the p-value is less than 0.05, the null hypothesis can be rejected, indicating the presence of heteroscedasticity.
f-value：This is the test statistic based on the F test. It measures the degree of heteroscedasticity. Like the Lagrange multiplier statistic, this value is primarily used in conjunction with its corresponding p-value.
f p-value：This is the p-value associated with the f-value. It is similar to the p-value interpretation of the Lagrange multiplier statistic.
'''


''' Check Residuals Distribution '''
# Kolmogorov-Smirnov Test
ks_statistic, ks_p_value = stats.kstest(model.resid, 'norm')
print(f"Kolmogorov-Smirnov Test:")
print(f"  Statistic: {ks_statistic:.4f}")
print(f"  p-value: {ks_p_value:.4f}")
# Short interpretation of results
print("\nShort interpretation of results：")
print("If the p-value is less than 0.05, the null hypothesis can be rejected, indicating that the residuals are not normally distributed.")
print("If the p-value is greater than 0.05, the null hypothesis cannot be rejected, indicating that the residuals are normally distributed.")
# Long interpretation of results
print("\nLong interpretation of results：")
print("Statistic：This is the test statistic for the Kolmogorov-Smirnov test. It measures the maximum distance between the empirical cumulative distribution function (ECDF) of the residuals and the cumulative distribution function (CDF) of the normal distribution. The value itself does not have a direct interpretation, but is used in conjunction with the p-value.")
print("p-value：This is the p-value associated with the test statistic. It represents the probability of observing a test statistic as extreme as the one observed, assuming the null hypothesis (the residuals are normally distributed) is true. Typically, a significance level of 0.05 is used. If the p-value is less than 0.05, the null hypothesis can be rejected, indicating that the residuals are not normally distributed.")

# QQ Plot
plt.figure(figsize=(10, 6), dpi=200)
stats.probplot(model.resid, dist="norm", plot=plt)
plt.title('QQ Plot')
plt.show()

# Residuals Distribution
plt.figure(figsize=(10, 6), dpi=200)
plt.hist(model.resid, bins=30, edgecolor='black', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


''' Clean up results '''
# Save Model 1 variables' Coefficient, Std. Error, p-value, VIF, and print the table
coefficients = fixed_effects_results.params
std_errors = fixed_effects_results.std_errors
p_values = fixed_effects_results.pvalues

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_model_1.columns
vif_data["VIF"] = [variance_inflation_factor(X_model_1.values, i) for i in range(X_model_1.shape[1])]

# Merge results
results_df = pd.DataFrame({
    "Coefficient": coefficients,
    "Std. Error": std_errors,
    "p-value": p_values
}).reset_index().rename(columns={"index": "Variable"})

# Merge VIF
results_df = results_df.merge(vif_data, on="Variable")

# Round values to four decimal places
results_df = results_df.round(4)

# Add stars based on significance
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''

results_df['p-value'] = results_df['p-value'].apply(lambda x: f"{x:.4f}{significance_stars(x)}")

# Print the table and explain the stars
print(results_df.to_markdown(index=False))
print("\nExplanation of the stars：")
print("***：Significant at the 0.1% level")
print("**：Significant at the 1% level")
print("*：Significant at the 5% level")
print(".: Significant at the 10% level")
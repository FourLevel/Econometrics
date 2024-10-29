import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from linearmodels.panel import PanelOLS, RandomEffects


# Read Excel file
file_path = '銀行資料_丟迴歸分析_20240523.csv'
# file_path = '銀行資料_丟迴歸分析_20241029.csv'
data = pd.read_csv(file_path)

# Select variables
df = data[['公司代碼', '年', 'Z-score', '資產總額', '淨收益', '當季季底P/B', 
           '負債比率', '公司年齡', '董事總持股數%', '經理人總持股%', 
           '是否在金融危機時期', '是否在COVID時期']]
df.columns = ['company_code', 'year', 'z_score', 'total_assets', 'net_income', 'pb_ratio', 
              'debt_ratio', 'company_age', 'shareholders_percentage', 'managers_percentage', 
              'crisis_period', 'covid_period']

# Set Panel Data
df = df.set_index(['company_code', 'year'])

# Standardize quantitative variables and add them to df
df[['std_total_assets', 'std_net_income', 'std_pb_ratio', 'std_debt_ratio', 'std_company_age', 'std_shareholders_percentage', 'std_managers_percentage']] = df[['total_assets', 'net_income', 'pb_ratio', 'debt_ratio', 'company_age', 'shareholders_percentage', 'managers_percentage']].apply(lambda x: (x - x.mean()) / x.std())

# Set y and X
y = df['z_score']
X = df[['std_total_assets', 'std_net_income', 'std_pb_ratio', 'std_debt_ratio', 'std_company_age', 'std_shareholders_percentage', 'std_managers_percentage', 'crisis_period', 'covid_period']]

# Choose X variables
X_model_1 = df[['std_total_assets', 'std_pb_ratio', 'std_debt_ratio', 'std_company_age', 'std_managers_percentage', 'crisis_period']]


''' Check Linear Relationship '''
# Plot dependent variable and independent variables
plt.figure(figsize=(10, 10), dpi=100)
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
plt.figure(figsize=(10, 10), dpi=100)
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

if dw_statistic < 1:
    print("\nConclusion：Significant positive autocorrelation")
elif dw_statistic > 3:
    print("\nConclusion：Significant negative autocorrelation")
else:
    print("\nConclusion：No significant autocorrelation")


''' Perform Breusch-Pagan test '''
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

'''
Interpretation of results
Lagrange multiplier statistic：This is the test statistic for the Breusch-Pagan test. It measures the degree of heteroscedasticity in the model. The value itself does not have a direct interpretation, but is used in conjunction with the p-value.
p-value：This is the p-value associated with the test statistic. It represents the probability of observing a test statistic as extreme as the one observed, assuming the null hypothesis (no heteroscedasticity) is true. Typically, a significance level of 0.05 is used. If the p-value is less than 0.05, the null hypothesis can be rejected, indicating the presence of heteroscedasticity.
f-value：This is the test statistic based on the F test. It measures the degree of heteroscedasticity. Like the Lagrange multiplier statistic, this value is primarily used in conjunction with its corresponding p-value.
f p-value：This is the p-value associated with the f-value. It is similar to the p-value interpretation of the Lagrange multiplier statistic.
'''


''' Check Normal Distribution '''
# Kolmogorov-Smirnov Test
ks_statistic, ks_p_value = stats.kstest(model.resid, 'norm')
print(f"Kolmogorov-Smirnov Test:")
print(f"  Statistic: {ks_statistic:.4f}")
print(f"  p-value: {ks_p_value:.4f}")

# QQ Plot
plt.figure(figsize=(10, 6), dpi=100)
stats.probplot(model.resid, dist="norm", plot=plt)
plt.title('QQ Plot')
plt.show()

# Residuals Distribution
plt.figure(figsize=(10, 6), dpi=100)
plt.hist(model.resid, bins=30, edgecolor='black', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


''' Perform Hausman Test '''
# Fixed effects model
fixed_effects_model = PanelOLS.from_formula('y ~ std_total_assets + std_pb_ratio + std_debt_ratio + std_company_age + std_managers_percentage + crisis_period + EntityEffects', data=df)
fixed_effects_results = fixed_effects_model.fit()

# Random effects model
random_effects_model = RandomEffects.from_formula('y ~ std_total_assets + std_pb_ratio + std_debt_ratio + std_company_age + std_managers_percentage + crisis_period', data=df)
random_effects_results = random_effects_model.fit()

# Perform Hausman test
def hausman(fe, re):
    b_diff = fe.params - re.params
    b_diff_cov = fe.cov - re.cov
    chi2 = np.dot(np.dot(b_diff.T, np.linalg.inv(b_diff_cov)), b_diff)
    df = b_diff.size
    p_value = stats.chi2.sf(chi2, df)
    return chi2, df, p_value

chi2, df, p_value = hausman(fixed_effects_results, random_effects_results)

hausman_result = {
    'Hausman test statistic': chi2,
    'Degrees of freedom': df,
    'p-value': p_value,
    'Fixed effects coefficients': fixed_effects_results.params,
    'Random effects coefficients': random_effects_results.params
}

# Display Hausman test results
print(f"Hausman test statistic: {hausman_result['Hausman test statistic']}")
print(f"Degrees of freedom: {hausman_result['Degrees of freedom']}")
print(f"p-value: {hausman_result['p-value']}")

# Display fixed effects model summary
print(fixed_effects_results.summary)

'''
Interpretation of results
Hausman test statistic：This is the test statistic. The larger the value, the greater the difference between the estimates of the fixed effects model and the random effects model.
Degrees of freedom：This is the degrees of freedom for the test statistic, usually equal to the number of parameters.
p-value：This is the p-value associated with the test statistic. If the p-value is less than a certain significance level (0.05), it indicates that the data is more suitable for the fixed effects model.
Fixed effects coefficients：This is the estimated coefficients of the fixed effects model.
Random effects coefficients：This is the estimated coefficients of the random effects model.
'''


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

# Print the table
print(results_df.to_markdown(index=False))
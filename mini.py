import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(123)

data = pd.read_csv("C:\\Users\\HP\\Downloads\\insurance.csv")

data['charges'] = np.log(data['charges'])

# Convert specific columns to numeric before creating dummies
for col in ['age', 'bmi', 'children']:  
    data[col] = pd.to_numeric(data[col])

print(data.head())
print(data.describe())

plt.figure(figsize=(12, 6))
sns.scatterplot(x='age', y='charges', data=data, color='blue', alpha=0.5)
plt.title('Correlation between Charges and Age')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='bmi', y='charges', data=data, color='green', alpha=0.5)
plt.title('Correlation between Charges and BMI')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='sex', y='charges', data=data)
plt.title('Charges by Sex')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='children', y='charges', data=data)
plt.title('Charges by Number of Children')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='smoker', y='charges', data=data)
plt.title('Charges by Smoking Status')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='region', y='charges', data=data)
plt.title('Charges by Region')
plt.show()

data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
X = data.drop('charges', axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train = X_train.astype(float)
X_test = X_test.astype(float)

X_train_const = sm.add_constant(X_train)
model_0 = sm.OLS(y_train, X_train_const).fit()
print(model_0.summary())
X_test_const = sm.add_constant(X_test)
predictions_0 = model_0.predict(X_test_const)
rmse_0 = np.sqrt(mean_squared_error(y_test, predictions_0))
print(f"RMSE for first model: {rmse_0:.2f}")

significant_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
X_train_significant = sm.add_constant(X_train[significant_features])
model_1 = sm.OLS(y_train, X_train_significant).fit()
print(model_1.summary())

X_test_significant = sm.add_constant(X_test[significant_features])
predictions_1 = model_1.predict(X_test_significant)
rmse_1 = np.sqrt(mean_squared_error(y_test, predictions_1))
print(f"RMSE for second model: {rmse_1:.2f}")

print(f"RMSE for first model: {rmse_0:.2f}")
print(f"RMSE for second model: {rmse_1:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(predictions_1, y_test, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Prediction vs. Actual Charges')
plt.xlabel('Predicted Charges')
plt.ylabel('Actual Charges')
plt.show()

residuals = y_test - predictions_1
plt.figure(figsize=(10, 6))
sns.scatterplot(x=predictions_1, y=residuals, color='blue', alpha=0.7)
plt.axhline(0, linestyle='--', color='red')
plt.title('Residuals vs. Predicted Charges')
plt.xlabel('Predicted Charges')
plt.ylabel('Residuals')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=15, kde=True, color='blue')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.show()

def predict_charges(model, age, bmi, children, smoker, region):
    regions = ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
    smoker_col = 'smoker_yes'
    data_point = {
        'age': age, 'bmi': bmi, 'children': children, 
        smoker_col: 1 if smoker == 'yes' else 0
    }
    for reg in regions:
        data_point[reg] = 1 if region == reg.split('_')[1] else 0
    data_df = pd.DataFrame([data_point])
    data_df = sm.add_constant(data_df)
    return model.predict(data_df)[0]

bob = predict_charges(model_1, age=19, bmi=27.9, children=0, smoker='yes', region='northwest')
lisa = predict_charges(model_1, age=40, bmi=50, children=2, smoker='no', region='southeast')
john = predict_charges(model_1, age=30, bmi=31.2, children=0, smoker='no', region='northeast')

print(f"Health care charges for Bob: ${bob:.2f}")
print(f"Health care charges for Lisa: ${lisa:.2f}")
print(f"Health care charges for John: ${john:.2f}")
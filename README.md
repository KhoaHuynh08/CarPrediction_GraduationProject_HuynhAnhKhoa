# CarPricePrediction_GraduationProject_HuynhAnhKhoa
# About this Project
A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
The expected outcome is to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels.
# Table Of Content


# 1. Importing Libraries and Loading Data

```python
import pandas as pd # Import the pandas library

url = '/content/CarPrice_Assignment.csv'
df = pd.read_csv(url)
df.sample(10)
    # Display the DataFrame for the current sheet
    # Or perform any other operations on sheet_data
```
# 2. Data Inspection
## Summary of the DataFrame, including the data types of each column, the number of non-null values
```python
df.info()
```
RangeIndex: 205 entries, 0 to 204.

Data columns (total 26 columns).

Data types: float64(8), int64(8), object(10).

## Removes duplicate rows from the DataFrame
```python
df.drop_duplicates()
```
0

## Preview of the data's structure and values
```python
df.head()
```

## Check the data type of each column in the DataFrame
```python
df.dtypes
```
car_ID	int64

symboling	int64

CarName	object

fueltype	object

aspiration	object

doornumber	object

carbody	object

drivewheel	object

enginelocation	object

wheelbase	float64

carlength	float64

carwidth	float64

carheight	float64

curbweight	int64

enginetype	object

cylindernumber	object

enginesize	int64

fuelsystem	object

boreratio	float64

stroke	float64

compressionratio	float64

horsepower	int64

peakrpm	int64

citympg	int64

highwaympg	int64

price	float64

## Understanding the distribution of values
```python
df.nunique()
```

car_ID	205

symboling	6

CarName	147

fueltype	2

aspiration	2

doornumber	2

carbody	5

drivewheel	3

enginelocation	2

wheelbase	53

carlength	75

carwidth	44

carheight	49

curbweight	171

enginetype	7

cylindernumber	7

enginesize	44

fuelsystem	8

boreratio	38

stroke	37

compressionratio	32

horsepower	59

peakrpm	23

citympg	29

highwaympg	30

price	189

## Statistics for the numerical columns in the DataFrame: standard deviation, minimum, and maximum values
```pytho
df.describe()
```

## 
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Distribution of Numerical Features
numerical_features = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                      'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                      'peakrpm', 'citympg', 'highwaympg', 'price']

plt.figure(figsize=(12, 8))
for feature in numerical_features:
    plt.subplot(3, 5, numerical_features.index(feature) + 1)
    sns.histplot(data=df[feature], bins=20, kde=True)
    plt.title(feature)
plt.tight_layout()
plt.show()
```

# Price Analysis
plt.figure(figsize=(8, 6))
sns.histplot(data=df['price'], bins=20, kde=True)
plt.title('Distribution of Price')
plt.show()
png

# Define the list of categorical columns to analyze
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
axes = axes.ravel()  # Flatten the 2D array of axes

# Loop through each categorical column
for i, column in enumerate(categorical_columns):
    sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
    for container in axes[i].containers:
        axes[i].bar_label(container, color='black', size=10)
    axes[i].set_title(f'Count Plot of {column.capitalize()}')
    axes[i].set_xlabel(column.capitalize())
    axes[i].set_ylabel('Count')

# Adjust layout and show plots
plt.tight_layout()
plt.show()
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
<ipython-input-12-e6b82ca5deba>:11: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
png

n = 20  # Number of top car models to plot
top_car_models = df['CarName'].value_counts().head(n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_car_models.values, y=top_car_models.index)
plt.title(f'Top {n} Car Models by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Car Model')
plt.tight_layout()
plt.show()
png

# Calculate average price for each car model
avg_prices_by_car = df.groupby('CarName')['price'].mean().sort_values(ascending=False)

# Plot top N car models by average price
n = 20  # Number of top car models to plot
top_car_models = avg_prices_by_car.head(n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_car_models.values, y=top_car_models.index)
plt.title(f'Top {n} Car Models by Average Price')
plt.xlabel('Average Price')
plt.ylabel('Car Model')
plt.tight_layout()
plt.show()
png

# Categorical Feature vs. Price
plt.figure(figsize=(12, 8))
for feature in categorical_columns:
    plt.subplot(3, 3, categorical_columns.index(feature) + 1)
    sns.boxplot(data=df, x=feature, y='price')
    plt.title(f'{feature} vs. Price')
plt.tight_layout()
plt.show()
png

# Correlation Analysis
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
png

# Extract brand and model from CarName
df['brand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
df['model'] = df['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]))

# Define categorical and numerical columns
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                     'peakrpm', 'citympg', 'highwaympg']

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder # Importing the LabelEncoder class

# Encoding categorical variables
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Feature engineering
df['power_to_weight_ratio'] = df['horsepower'] / df['curbweight']
for column in numerical_columns:
    df[f'{column}_squared'] = df[column] ** 2
df['log_enginesize'] = np.log(df['enginesize'] + 1)

# Feature scaling
# Import StandardScaler
from sklearn.preprocessing import StandardScaler # Importing StandardScaler
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
# Extract brand and model from CarName
df['brand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
df['model'] = df['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]))

# Define categorical and numerical columns
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                     'peakrpm', 'citympg', 'highwaympg']

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder # Importing the LabelEncoder class

# Encoding categorical variables
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Feature engineering
df['power_to_weight_ratio'] = df['horsepower'] / df['curbweight']
for column in numerical_columns:
    df[f'{column}_squared'] = df[column] ** 2
df['log_enginesize'] = np.log(df['enginesize'] + 1)

# Feature scaling
# Import StandardScaler
from sklearn.preprocessing import StandardScaler # Importing StandardScaler
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py:399: RuntimeWarning: invalid value encountered in log
  result = getattr(ufunc, method)(*inputs, **kwargs)
# Import SimpleImputer if not already imported
from sklearn.impute import SimpleImputer

# Create and fit an imputer
imputer = SimpleImputer(strategy='mean')  # Or other strategies like 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)  # Use the same imputer fitted on training data

# Model training with imputed data
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Make predictions using imputed data
y_pred = model.predict(X_test_imputed)
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

<ipython-input-26-d1328c60a5c7> in <cell line: 0>()
      4 # Create and fit an imputer
      5 imputer = SimpleImputer(strategy='mean')  # Or other strategies like 'median', 'most_frequent'
----> 6 X_train_imputed = imputer.fit_transform(X_train)
      7 X_test_imputed = imputer.transform(X_test)  # Use the same imputer fitted on training data
      8 


NameError: name 'X_train' is not defined
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Assuming 'df' is your DataFrame and 'price' is your target variable
X = df[['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model',
       'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
       'peakrpm', 'citympg', 'highwaympg']]  # Select your features
y = df['price']  # Select your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splitting the data

# Create and fit an imputer
imputer = SimpleImputer(strategy='mean')  # Or other strategies like 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)  # Use the same imputer fitted on training data

# Model training with imputed data
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Make predictions using imputed data
y_pred = model.predict(X_test_imputed)
import statsmodels.api as sm
X = df[['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model','wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                     'peakrpm', 'citympg', 'highwaympg']]
Y = df['price']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.903
Model:                            OLS   Adj. R-squared:                  0.890
Method:                 Least Squares   F-statistic:                     69.49
Date:                Sun, 23 Mar 2025   Prob (F-statistic):           4.94e-78
Time:                        13:43:51   Log-Likelihood:                -1893.8
No. Observations:                 205   AIC:                             3838.
Df Residuals:                     180   BIC:                             3921.
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const             1.203e+04   5955.141      2.020      0.045     278.162    2.38e+04
fueltype          3311.7353   6451.384      0.513      0.608   -9418.334     1.6e+04
aspiration         872.7411    880.170      0.992      0.323    -864.037    2609.519
doornumber        -586.0138    574.672     -1.020      0.309   -1719.974     547.947
carbody          -1043.7132    358.245     -2.913      0.004   -1750.612    -336.814
drivewheel         835.9231    535.802      1.560      0.120    -221.338    1893.184
enginelocation    1.178e+04   2051.525      5.742      0.000    7731.538    1.58e+04
enginetype         216.3884    221.986      0.975      0.331    -221.642     654.419
cylindernumber      21.6941    351.514      0.062      0.951    -671.925     715.313
fuelsystem         121.3683    147.913      0.821      0.413    -170.499     413.236
brand             -172.6777     31.337     -5.510      0.000    -234.512    -110.843
model               18.1761      5.681      3.200      0.002       6.967      29.385
wheelbase          883.8934    580.832      1.522      0.130    -262.222    2030.009
carlength         -268.1113    635.405     -0.422      0.674   -1521.911     985.689
carwidth          1408.7918    547.415      2.574      0.011     328.617    2488.967
carheight          760.4582    328.087      2.318      0.022     113.067    1407.850
curbweight        1748.2716    821.942      2.127      0.035     126.390    3370.153
enginesize        3019.2148    714.594      4.225      0.000    1609.157    4429.273
boreratio         -123.4494    297.036     -0.416      0.678    -709.570     462.671
stroke            -880.6062    233.252     -3.775      0.000   -1340.866    -420.347
compressionratio  1335.3595   1831.998      0.729      0.467   -2279.596    4950.315
horsepower         901.4236    709.160      1.271      0.205    -497.912    2300.759
peakrpm            889.8598    311.348      2.858      0.005     275.498    1504.221
citympg           -631.7534   1044.652     -0.605      0.546   -2693.092    1429.586
highwaympg         984.1344    977.763      1.007      0.316    -945.217    2913.486
==============================================================================
Omnibus:                       32.131   Durbin-Watson:                   0.946
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              137.420
Skew:                           0.469   Prob(JB):                     1.44e-30
Kurtosis:                       6.900   Cond. No.                     3.78e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.78e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
import statsmodels.api as sm
X = df[['enginelocation', 'brand', 'model', 'carwidth', 'curbweight',
                     'enginesize', 'stroke',
                     'peakrpm']]
Y = df['price']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.886
Model:                            OLS   Adj. R-squared:                  0.881
Method:                 Least Squares   F-statistic:                     190.3
Date:                Sun, 23 Mar 2025   Prob (F-statistic):           4.55e-88
Time:                        13:43:57   Log-Likelihood:                -1909.9
No. Observations:                 205   AIC:                             3838.
Df Residuals:                     196   BIC:                             3868.
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
const           1.434e+04    598.205     23.974      0.000    1.32e+04    1.55e+04
enginelocation  1.486e+04   1829.256      8.121      0.000    1.12e+04    1.85e+04
brand           -152.7134     27.676     -5.518      0.000    -207.294     -98.133
model             13.9152      5.479      2.540      0.012       3.109      24.721
carwidth        1977.5460    404.106      4.894      0.000    1180.592    2774.500
curbweight      2519.6339    512.616      4.915      0.000    1508.682    3530.585
enginesize      3083.7259    433.194      7.119      0.000    2229.406    3938.045
stroke          -791.1407    203.421     -3.889      0.000   -1192.315    -389.966
peakrpm          690.6051    212.703      3.247      0.001     271.124    1110.086
==============================================================================
Omnibus:                       25.685   Durbin-Watson:                   0.826
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               63.522
Skew:                           0.532   Prob(JB):                     1.61e-14
Kurtosis:                       5.511   Cond. No.                         753.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
import statsmodels.api as sm
X = df[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                     'peakrpm', 'citympg', 'highwaympg']]
Y = df['price']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.851
Model:                            OLS   Adj. R-squared:                  0.841
Method:                 Least Squares   F-statistic:                     83.78
Date:                Sun, 23 Mar 2025   Prob (F-statistic):           1.68e-71
Time:                        13:44:01   Log-Likelihood:                -1937.5
No. Observations:                 205   AIC:                             3903.
Df Residuals:                     191   BIC:                             3949.
Df Model:                          13                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const             1.328e+04    222.741     59.606      0.000    1.28e+04    1.37e+04
wheelbase          736.5685    603.500      1.220      0.224    -453.811    1926.948
carlength        -1165.1827    683.747     -1.704      0.090   -2513.848     183.482
carwidth          1081.9057    526.460      2.055      0.041      43.483    2120.329
carheight          397.7606    330.827      1.202      0.231    -254.783    1050.305
curbweight         978.8891    902.401      1.085      0.279    -801.063    2758.842
enginesize        4874.6737    574.784      8.481      0.000    3740.934    6008.414
boreratio         -270.8754    323.083     -0.838      0.403    -908.145     366.394
stroke            -949.3195    243.572     -3.897      0.000   -1429.755    -468.884
compressionratio  1181.3201    328.535      3.596      0.000     533.297    1829.343
horsepower        1215.3251    639.694      1.900      0.059     -46.446    2477.096
peakrpm           1130.1391    319.210      3.540      0.001     500.510    1759.768
citympg          -2090.6868   1160.153     -1.802      0.073   -4379.044     197.670
highwaympg        1393.3118   1097.494      1.270      0.206    -771.454    3558.078
==============================================================================
Omnibus:                       24.541   Durbin-Watson:                   0.930
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               81.326
Skew:                           0.383   Prob(JB):                     2.19e-18
Kurtosis:                       5.989   Cond. No.                         18.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
import statsmodels.api as sm
X = df[['carwidth', 'enginesize', 'stroke', 'compressionratio', 'peakrpm', ]]
Y = df['price']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.835
Model:                            OLS   Adj. R-squared:                  0.830
Method:                 Least Squares   F-statistic:                     200.9
Date:                Sun, 23 Mar 2025   Prob (F-statistic):           1.01e-75
Time:                        13:44:05   Log-Likelihood:                -1948.0
No. Observations:                 205   AIC:                             3908.
Df Residuals:                     199   BIC:                             3928.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const             1.328e+04    229.741     57.790      0.000    1.28e+04    1.37e+04
carwidth          1923.9440    349.340      5.507      0.000    1235.060    2612.828
enginesize        6113.3947    354.828     17.229      0.000    5413.690    6813.100
stroke           -1019.4106    239.313     -4.260      0.000   -1491.325    -547.496
compressionratio   883.5738    268.423      3.292      0.001     354.255    1412.892
peakrpm           1555.1961    265.168      5.865      0.000    1032.297    2078.095
==============================================================================
Omnibus:                       25.980   Durbin-Watson:                   0.902
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               67.376
Skew:                           0.521   Prob(JB):                     2.34e-15
Kurtosis:                       5.608   Cond. No.                         2.97
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# Evaluate the model
# Import necessary functions
from sklearn.metrics import mean_squared_error, r2_score # Importing the required metrics

mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test, y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')
 R-squared: 0.8438286849668024
Mean Squared Error: 12328791.554784345
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
pred_df
Actual Value	Predicted Value	Difference
15	30760.000	25937.522556	4822.477444
9	17859.167	17702.472395	156.694605
100	9549.000	10137.661869	-588.661869
132	11850.000	13269.164910	-1419.164910
68	28248.000	26333.431539	1914.568461
95	7799.000	6713.930205	1085.069795
159	7788.000	8019.618092	-231.618092
162	9258.000	6309.100244	2948.899756
147	10198.000	11188.100762	-990.100762
182	7775.000	8006.626939	-231.626939
191	13295.000	15083.266849	-1788.266849
164	8238.000	5776.309083	2461.690917
65	18280.000	16817.349541	1462.650459
175	9988.000	10185.823912	-197.823912
73	40960.000	38710.667606	2249.332394
152	6488.000	5924.456933	563.543067
18	5151.000	-931.407501	6082.407501
82	12629.000	14656.758063	-2027.758063
86	8189.000	9920.816179	-1731.816179
143	9960.000	10172.254258	-212.254258
60	8495.000	11179.832717	-2684.832717
101	13499.000	20656.064078	-7157.064078
98	8249.000	7400.970927	848.029073
30	6479.000	1759.296369	4719.703631
25	6692.000	6965.819355	-273.819355
16	41315.000	25336.495911	15978.504089
168	9639.000	13477.320346	-3838.320346
195	13415.000	15514.943636	-2099.943636
97	7999.000	5544.841654	2454.158346
194	12940.000	16011.069531	-3071.069531
67	25552.000	26814.324359	-1262.324359
120	6229.000	7140.003604	-911.003604
154	7898.000	4511.861504	3386.138496
202	21485.000	21347.993607	137.006393
79	7689.000	8523.138061	-834.138061
69	28176.000	26636.236853	1539.763147
145	11259.000	11343.957100	-84.957100
55	10945.000	8797.784865	2147.215135
45	8916.500	5764.619697	3151.880303
84	14489.000	14798.179719	-309.179719
146	7463.000	8564.777050	-1101.777050
<script>
  const buttonEl =
    document.querySelector('#df-c4cea1b1-caf9-42cc-8d41-f940947aa8e2 button.colab-df-convert');
  buttonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
    const element = document.querySelector('#df-c4cea1b1-caf9-42cc-8d41-f940947aa8e2');
    const dataTable =
      await google.colab.kernel.invokeFunction('convertToInteractive',
                                                [key], {});
    if (!dataTable) return;

    const docLinkHtml = 'Like what you see? Visit the ' +
      '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
      + ' to learn more about interactive tables.';
    element.innerHTML = '';
    dataTable['output_type'] = 'display_data';
    await google.colab.output.renderOutput(dataTable, element);
    const docLink = document.createElement('div');
    docLink.innerHTML = docLinkHtml;
    element.appendChild(docLink);
  }
</script>

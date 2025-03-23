# CarPricePrediction_GraduationProject_HuynhAnhKhoa
# About this Project
A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.

The expected outcome is to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels.

Source: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/code

# Table Of Content
# 1. Importing Libraries and Loading Data
# 2. Data Inspection
### Summary of the DataFrame, including the data types of each column, the number of non-null values
### Removes duplicate rows from the DataFrame
### Preview of the data's structure and values
### Check the data type of each column in the DataFrame
### Understanding the distribution of values
### Statistics for the numerical columns in the DataFrame: standard deviation, minimum, and maximum values
### Analyzing Categorical Data
# 3. Visualizing Numerical Data Distributions
### Distribution of Numerical Features
### Price Analysis
### Define the list of categorical columns to analyze
### Number of top car models by Frequency
### Top car models by average price
### Categorical Feature vs. Price
# 4. Correlation Analysis
# 5. Linear Regression Modeling


# 1. Importing Libraries and Loading Data
Source: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/code
*Data set of different types of cars across the America market.
Description: The dataset includes different types of cars, product information, and specifications.

Structure:

Car_ID		Unique id of each observation (Interger)                  	

Symboling	Its assigned insurance risk rating, A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.(Categorical)                         	

carCompany	Name of car company (Categorical)                           	

fueltype	Car fuel type i.e gas or diesel (Categorical)                            	

aspiration	Aspiration used in a car (Categorical)                        	

doornumber	Number of doors in a car (Categorical)                     	

carbody	body of car (Categorical)                    	

drivewheel 	type of drive wheel (Categorical)                   	

enginelocation Location of car engine (Categorical)                           	

wheelbase 	Weelbase of car (Numeric)               	

carlength 	Length of car (Numeric)                      	

carwidth 	Width of car (Numeric)                        	

carheight 	height of car (Numeric)                        	

curbweight 	The weight of a car without occupants or baggage. (Numeric) 	            	

enginetype 	Type of engine. (Categorical)                           	

cylindernumber cylinder placed in the car (Categorical)                    	

enginesize 	Size of car (Numeric)            	

fuelsystem 	Fuel system of car (Categorical)                    	

boreratio 	Boreratio of car (Numeric)                 	

stroke 		Stroke or volume inside the engine (Numeric)                     	

compressionratio 	compression ratio of car (Numeric)            	

horsepower 	Horsepower (Numeric)                        	

peakrpm 	car peak rpm (Numeric)                      	

citympg 	Mileage in city (Numeric)                   	

highwaympg 	Mileage on highway (Numeric)                       	

price(Dependent variable)	Price of car (Numeric)


```python
import pandas as pd # Import the pandas library

url = '/content/CarPrice_Assignment.csv'
df = pd.read_csv(url)
df.sample(10)
    # Display the DataFrame for the current sheet
    # Or perform any other operations on sheet_data
```
![image](https://github.com/user-attachments/assets/69f5d193-faf6-4906-90a0-cf7a92f2ba0e)

# 2. Data Inspection
## Summary of the DataFrame, including the data types of each column, the number of non-null values
```python
df.info()
```
RangeIndex: 205 entries, 0 to 204.

Data columns (total 26 columns).

Data types: float64(8), int64(8), object(10).
![image](https://github.com/user-attachments/assets/0c7c689c-64bc-4e02-8dc3-1f48ba54f695)


## Removes duplicate rows from the DataFrame
```python
df.drop_duplicates()
```
0
![image](https://github.com/user-attachments/assets/8dcdd595-7f67-4f18-9700-b28b10af0732)


## Preview of the data's structure and values
```python
df.head()
```
![image](https://github.com/user-attachments/assets/0c7c38a4-d01e-4184-8268-e7df162a5f91)

## Check the data type of each column in the DataFrame
```python
df.dtypes
```
![image](https://github.com/user-attachments/assets/08501763-49fe-4630-9b05-6253990d93a7)


## Understanding the distribution of values
```python
df.nunique()
```
![image](https://github.com/user-attachments/assets/bfbb1658-4898-4346-ad65-3572e226deb0)


## Statistics for the numerical columns in the DataFrame: standard deviation, minimum, and maximum values
```pytho
df.describe()
```
![image](https://github.com/user-attachments/assets/bb465068-8b7f-4ed8-9d57-66e13fabff6f)

## Analyzing Categorical Data
This helps to see the distinct categories within each categorical feature.

```python
categorical_columns = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype',
    'cylindernumber',
    'fuelsystem']

for col in categorical_columns:

    print(f"Category in {col} is : {df[col].unique()}")
```
![image](https://github.com/user-attachments/assets/327b6a1f-c3b1-4123-abbe-bc99853c238f)


# 3. Visualizing Numerical Data Distributions
## Distribution of Numerical Features
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
![image](https://github.com/user-attachments/assets/016a2f09-eaff-45dd-b966-1b83e25aa716)


## Price Analysis
```python
plt.figure(figsize=(8, 6))
sns.histplot(data=df['price'], bins=20, kde=True)
plt.title('Distribution of Price')
plt.show()
```
![image](https://github.com/user-attachments/assets/2aaf65bc-b111-459e-ba02-f90623209504)


## Define the list of categorical columns to analyze
```python
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
#Create subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
axes = axes.ravel()  # Flatten the 2D array of axes

#Loop through each categorical column
for i, column in enumerate(categorical_columns):
    sns.countplot(x=df[column], data=df, palette='bright', ax=axes[i], saturation=0.95)
    for container in axes[i].containers:
        axes[i].bar_label(container, color='black', size=10)
    axes[i].set_title(f'Count Plot of {column.capitalize()}')
    axes[i].set_xlabel(column.capitalize())
    axes[i].set_ylabel('Count')

#Adjust layout and show plots
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/bf63cade-b003-4a92-a6d8-b468be58ef7a)
![image](https://github.com/user-attachments/assets/47e81bc3-839f-41cd-bca6-809aab04675b)
![image](https://github.com/user-attachments/assets/9fc2db98-f27f-4c4c-ad98-7522bb5c9ab2)

## Number of top car models by Frequency
```python

n = 20  # Number of top car models to plot
top_car_models = df['CarName'].value_counts().head(n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_car_models.values, y=top_car_models.index)
plt.title(f'Top {n} Car Models by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Car Model')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/c0d5576d-b0f6-4b66-9ca8-154c6761e8e9)

## Top car models by average price
```python
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
```
![image](https://github.com/user-attachments/assets/5d502ab2-f1dc-4b38-a27f-47b2aacefd10)


## Categorical Feature vs. Price
```python
plt.figure(figsize=(12, 8))
for feature in categorical_columns:
    plt.subplot(3, 3, categorical_columns.index(feature) + 1)
    sns.boxplot(data=df, x=feature, y='price')
    plt.title(f'{feature} vs. Price')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/c3596c18-2ea4-463e-afb0-b9311bca4d44)
![image](https://github.com/user-attachments/assets/42d83d64-d98a-4342-9019-aedbda495e3b)



# 4. Correlation Analysis
```python
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```
![image](https://github.com/user-attachments/assets/80fb6548-8bb1-4faf-884c-476da7230ef5)


# 5. Linear Regression Modeling
```python
import statsmodels.api as sm
X = df[['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model','wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                     'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                     'peakrpm', 'citympg', 'highwaympg']]
Y = df['price']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())
```
![image](https://github.com/user-attachments/assets/7e39a834-a0d9-4b2c-b032-530e552242ab)
![image](https://github.com/user-attachments/assets/380843de-e52b-4c39-a071-e0540532363a)


"""Untitled4.ipynb
Original file is located at
    https://colab.research.google.com/drive/1y1Av_iXaN-7145XgTfj_bsXmgktLdln7
"""

# 1 Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset into a pandas DataFrame
# Assuming the dataset is in a CSV file named 'gold_prices.csv'
df = pd.read_csv('data_csv.csv')

# 2 Preprocess the data
# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
# Extract year and month from the date column and create separate columns for them
df['year'] = df['Date'].dt.year
# Drop the original date column as we only need year and month for prediction
df = df.drop('Date', axis=1)

# 3 Split the data into training and testing setsStep 4: Split the data into training and testing sets
X = df[['year']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 4 Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# 5 Make predictions for a specific year (ask for prediction year)
prediction_year = int(input("Enter the year for which you want to predict gold prices: "))
predicted_price = model.predict([[prediction_year]])
print(f"The predicted gold price for {prediction_year} is ${predicted_price[0]:.2f}")

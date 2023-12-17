# Import libraries
import tkinter as tk
import tkinter.ttk as ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create the main window
window = tk.Tk()
window.title("Gold Price Predictor")
window.minsize(700, 190)
window.maxsize(700, 190)


def on_button_click():

    # Read the File
    df = pd.read_csv('gold_prices.csv')

    # Preprocess the data

    # Convert the date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract year and month from the date column and create separate columns for them
    df['year'] = df['Date'].dt.year

    # Drop the original date column as we only need year and month for prediction
    df = df.drop('Date', axis=1)

    # Split the data into training and testing sets
    X = df[['year']]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction_year = int(a1.get())
    predicted_price = model.predict([[prediction_year]])

    label.config(text=f"The predicted gold price for {prediction_year} is ${predicted_price[0]:.2f}")


# Label
label = ttk.Label(window, text="Enter the year for which you want to predict gold prices:")
label.place(x=140, y=60)

# Entry
a1 = tk.ttk.Entry(window)
a1.place(x=440, y=60)

# Button
button = ttk.Button(window, text="Predict", command=on_button_click)
button.place(x=280, y=120)

# Main loop
window.mainloop()
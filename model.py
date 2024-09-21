# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


# Load your dataset
# Make sure to replace 'DataCenterCoolingSampleData.csv' with the actual path to your CSV file
df = pd.read_csv('DataCenterCoolingData.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Separate features (X) and target (y)
X = df[['Number of Servers', 'Season', 'Server Time (hours)', 'Surrounding Temperature (°C)', 'Current Server Temperature (°C)', 'Maintainable Server Temperature (°C)']]
y = df['Total Water Required (liters)']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# make pickle file of  our model
pickle.dump(model, open("model.pkl", "wb"))
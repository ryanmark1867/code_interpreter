# code interpreter output for prompt
# share code to train a Keras model on this dataset where the trained model would be able to predict whether a 
# given listing would have a price above or below the average price
# based on CSV file at https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# NEWLY ADDED POST CODE INTERPRETER OUTPUT
import pandas as pd
data = pd.read_csv('AB_NYC_2019.csv')

# Define the target variable (above or below the average price)
average_price = data['price'].mean()
data['above_average'] = (data['price'] > average_price).astype(int)

# Select the features and target
features = ['neighbourhood_group', 'room_type', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']
target = 'above_average'

# Perform label encoding for categorical features
for feature in ['neighbourhood_group', 'room_type']:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
print("average price is: ",average_price)
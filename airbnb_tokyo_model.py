# code interpreter output on tokyo listing.csv dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# import dataset
import pandas as pd

# Load the data
data = pd.read_csv('listings.csv')


# Calculate the average price
average_price = data['price'].mean()

# Create a new binary column indicating whether the price is above average
data['price_above_average'] = (data['price'] > average_price).astype(int)

# Select the columns of interest
selected_columns = ['room_type', 'bedrooms', 'bathrooms', 'accommodates', 'number_of_reviews', 'review_scores_rating', 'amenities', 'price_above_average']
data_selected = data[selected_columns]

# Drop rows with missing values
data_selected = data_selected.dropna()

# Reset the index of the selected data
data_selected.reset_index(drop=True, inplace=True)

# One-hot encoding for the 'room_type' column
one_hot_encoder = OneHotEncoder(sparse=False)
room_type_encoded = one_hot_encoder.fit_transform(data_selected[['room_type']])
room_type_encoded_df = pd.DataFrame(room_type_encoded, columns=one_hot_encoder.get_feature_names(['room_type']))

# Text processing for the 'amenities' column
count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
amenities_count = count_vectorizer.fit_transform(data_selected['amenities']).toarray()
amenities_count_df = pd.DataFrame(amenities_count, columns=count_vectorizer.get_feature_names())

# Combine the processed features
processed_data = pd.concat([data_selected.drop(['room_type', 'amenities'], axis=1), room_type_encoded_df, amenities_count_df], axis=1)

# Split the data into input features (X) and target variable (y)
X = processed_data.drop('price_above_average', axis=1)
y = processed_data['price_above_average']

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Display the shapes of the training and test sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# model part

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

# Print the accuracy of the model
print("Accuracy: ", scores[1])

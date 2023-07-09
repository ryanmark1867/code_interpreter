# code interpreter output for prompt:
#
# share code to train a Keras model on this dataset where the trained model would be able to 
# predict whether a given listing would have a price above or below the average price.
#
# based on CSV file at https://www.kaggle.com/datasets/dragonduck/property-listings-in-kuala-lumpur


import pandas as pd

# Load the dataset
df = pd.read_csv('kl_data_kaggle.csv')

# Show the first few rows of the dataframe
df.head()

# Data cleanup (iteration 3)

# Correctly preprocess 'Rooms' column - treating 2+1 as 2.1 rooms, 'Studio' as 1 room and '20 Above' as 20, for instance
df['Rooms'] = df['Rooms'].replace({'Studio': '1', '20 Above': '20'}).str.replace('+', '.').astype(float)

# Fill missing values with median for numerical columns
for col in ['Price', 'Rooms', 'Bathrooms', 'Car Parks', 'Size']:
    df[col] = df[col].fillna(df[col].median())

# Fill missing values with mode (most frequent value) for categorical columns
for col in ['Location', 'Property Type', 'Furnishing']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Create target column 'Above_Average_Price'
average_price = df['Price'].mean()
df['Above_Average_Price'] = (df['Price'] > average_price).astype(int)

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
df_encoded = pd.DataFrame(encoder.fit_transform(df[['Location', 'Property Type', 'Furnishing']]))
df_encoded.columns = encoder.get_feature_names(['Location', 'Property Type', 'Furnishing'])

# Drop original categorical columns and merge with one-hot encoded dataframe
df.drop(['Location', 'Property Type', 'Furnishing'], axis=1, inplace=True)
df = pd.concat([df, df_encoded], axis=1)

# model training

# Import necessary libraries
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define X and y
X = df.drop('Above_Average_Price', axis=1)
y = df['Above_Average_Price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Keras model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10)

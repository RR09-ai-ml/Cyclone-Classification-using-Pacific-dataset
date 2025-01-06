import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load the dataset
file_path = 'pacific_data/p1.csv'
df = pd.read_csv(file_path)

# Preprocess data for efficient training. (Convert categorical labels to numerical values)
label_encoder = LabelEncoder()
df['Cyclone Intensity'] = label_encoder.fit_transform(df['Cyclone Intensity'])
df['Latitude'] = label_encoder.fit_transform(df['Latitude'])
df['Longitude'] = label_encoder.fit_transform(df['Longitude'])

# Add latitude and longitude to features
features = [
    'Latitude', 'Longitude',  # Added features (get error because in string format (convert to numeric format))
    'Max Wind Speed', 'Central Pressure', 
    'Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW', 
    'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW', 
    'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW'
]

X = df[features]
y = df['Cyclone Intensity']

# Split the data into training and testing sets (0.2 means 20% data for test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1]

# Build and compile the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_shape,)))  # ReLU (Rectified Linear Unit)
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))  # 7 unique categories for cyclone intensity
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model and print the classification report
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
print("Classification Report:")
# print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
print(classification_report(y_test, y_pred_classes, labels=np.unique(y_test), target_names=label_encoder.classes_[:7]))

print(f"Accuracy: {accuracy_score(y_test, y_pred_classes):.2f}")

# # Save the model
# model.save('cyclone_pridict.keras')

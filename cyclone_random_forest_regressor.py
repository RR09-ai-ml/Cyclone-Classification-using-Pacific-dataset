import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# Load the dataset
file_path = 'Cyclone Classification using Pacific dataset/pacific.csv/pacific.csv'
# file_path = 'atlantic.csv/atlantic.csv'
data = pd.read_csv(file_path)
print(data.head(10))


# Data Preprocessing
# Convert latitude and longitude to numeric (remove 'N', 'S', 'E', 'W')
data['Latitude'] = data['Latitude'].str.replace(r'[^\d.]', '', regex=True).astype(float)
data['Longitude'] = data['Longitude'].str.replace(r'[^\d.]', '', regex=True).astype(float)


# Show the count of missing values and fill them with mean.
for column in data.columns:
    missing_cnt = data[column][data[column] == -999].count()
    print('Missing Values in column {col} = '.format(col = column) , missing_cnt )
    if missing_cnt!= 0:
#         print('in ' , column)
        mean = round(data[column][data[column] != -999 ].mean())
#         print("mean",mean)
        index = data.loc[data[column] == -999 , column].index
#         print("index" , index )
        data.loc[data[column] == -999 , column] = mean
#         print(df.loc[index , column])


# after roundof and mean of missing values
print(data.head(10))


# Encode categorical variables like 'Event' and 'Status'
label_encoder = LabelEncoder()
data['Event'] = label_encoder.fit_transform(data['Event'])
data['Status'] = label_encoder.fit_transform(data['Status'])


# Feature Selection: Choose relevant features for prediction
features = data[['Latitude', 'Longitude', 'Minimum Pressure', 'Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW', 'Event', 'Status']]
target = data['Maximum Wind']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Selection: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse}")


# Model Performance: You can also evaluate using other metrics like R-squared, MAE, etc.
from sklearn.metrics import r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print("\nR squared is a accuracy for regression model.")
# print("It indicates how well the model's predictions fit the actual data\n \
#             R² = 1: Perfect prediction.\n \
#             R² = 0: The model doesn't explain any variance (as good as a simple mean predictor).\n \
#             Negative R²: Indicates that the model performs worse than predicting the mean of the target.\n")

print(f"R-squared: {r2}")

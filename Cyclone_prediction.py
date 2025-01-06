import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Convert categorical labels to numerical values
    label_encoder = LabelEncoder()   
    df['Cyclone Intensity'] = label_encoder.fit_transform(df['Cyclone Intensity'])

    # Select features and target variable
    features = [
        'Max Wind Speed', 'Central Pressure', 
        'Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW', 
        'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW', 
        'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW'
    ]
    
    X = df[features]
    y = df['Cyclone Intensity']
    
    return X, y, label_encoder

def train_model(X_train, y_train):
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    # Evaluate the model and print the classification report.
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


if __name__ == "__main__":
    # Load data
    # file_path = input("Enter the name of the CSV file (e.g., p1.csv): ")
    file_path = 'Cyclone Classification using Pacific dataset/pacific_data/p1.csv'
    df = load_data(file_path)

    # Preprocess data
    X, y, label_encoder = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, label_encoder)



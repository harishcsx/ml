import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(url):
    features = {}
    
    # URL length
    features['length'] = len(url)
    
    # Count of special characters
    features['special_chars'] = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))
    
    # Presence of 'login' keyword
    features['has_login'] = int('login' in url)
    
    # Use of HTTPS
    features['is_https'] = int(urlparse(url).scheme == 'https')
    
    return features

def load_and_prepare_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Extract features
    features = df['URL'].apply(extract_features)
    features_df = pd.DataFrame(features.tolist())
    
    # Combine features with labels
    X = features_df
    y = df['Label']
    
    return X, y

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    return model

def classify_url(model, url):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)[0]
    
    if prediction == 1:
        print("Suspicious")
    else:
        print("Genuine")

if __name__ == "__main__":
    # Update the path to your dataset file if necessary
    file_path = 'phishing_dataset.csv'  # Update this if the file is in a different location
    X, y = load_and_prepare_data(file_path)
    
    # Train the model
    model = train_model(X, y)
    
    # Prompt user for a URL
    user_url = input("Enter the URL to be classified: ")
    
    # Classify the user-provided URL
    classify_url(model, user_url)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(row):
    features = {
        'URLLength': row['URLLength'],
        'DomainLength': row['DomainLength'],
        'IsDomainIP': row['IsDomainIP'],
        'URLSimilarityIndex': row['URLSimilarityIndex'],
        'CharContinuationRate': row['CharContinuationRate'],
        'TLDLegitimateProb': row['TLDLegitimateProb'],
        'TLDLength': row['TLDLength'],
        'NoOfSubDomain': row['NoOfSubDomain'],
        'HasObfuscation': row['HasObfuscation'],
        'NoOfLettersInURL': row['NoOfLettersInURL'],
        'LetterRatioInURL': row['LetterRatioInURL'],
        'NoOfDigitsInURL': row['NoOfDegitsInURL'],
        'DigitRatioInURL': row['DegitRatioInURL'],
        'NoOfSpecialCharsInURL': row['NoOfOtherSpecialCharsInURL'],
        'SpacialCharRatioInURL': row['SpacialCharRatioInURL'],
        'IsHTTPS': row['IsHTTPS'],
        'LineOfCode': row['LineOfCode'],
        'LargestLineLength': row['LargestLineLength'],
        'DomainTitleMatchScore': row['DomainTitleMatchScore'],
        'URLTitleMatchScore': row['URLTitleMatchScore'],
        'NoOfURLRedirect': row['NoOfURLRedirect'],
        'NoOfSelfRedirect': row['NoOfSelfRedirect'],
        'HasDescription': row['HasDescription'],
        'HasExternalFormSubmit': row['HasExternalFormSubmit'],
        'HasSocialNet': row['HasSocialNet'],
        'HasSubmitButton': row['HasSubmitButton'],
        'HasHiddenFields': row['HasHiddenFields'],
        'HasPasswordField': row['HasPasswordField'],
        'NoOfImage': row['NoOfImage'],
        'NoOfCSS': row['NoOfCSS'],
        'NoOfJS': row['NoOfJS'],
        'NoOfSelfRef': row['NoOfSelfRef'],
        'NoOfEmptyRef': row['NoOfEmptyRef'],
        'NoOfExternalRef': row['NoOfExternalRef'],
    }
    return features

def load_and_prepare_data(file_path):
    # Load the dataset from CSV
    df = pd.read_csv(file_path)
    
    # Extract features
    features = df.apply(extract_features, axis=1)
    features_df = pd.DataFrame(features.tolist())
    
    # Separate features and labels
    X = features_df
    y = df['label']
    
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

def classify_url(model, url, **kwargs):
    # Classify a new URL based on input features
    features = {
        'URLLength': len(url),
        'DomainLength': kwargs.get('DomainLength', 0),
        'IsDomainIP': kwargs.get('IsDomainIP', 0),
        'URLSimilarityIndex': kwargs.get('URLSimilarityIndex', 100),
        'CharContinuationRate': kwargs.get('CharContinuationRate', 1),
        'TLDLegitimateProb': kwargs.get('TLDLegitimateProb', 0.5),
        'TLDLength': kwargs.get('TLDLength', 3),
        'NoOfSubDomain': kwargs.get('NoOfSubDomain', 0),
        'HasObfuscation': kwargs.get('HasObfuscation', 0),
        'NoOfLettersInURL': kwargs.get('NoOfLettersInURL', 0),
        'LetterRatioInURL': kwargs.get('LetterRatioInURL', 0),
        'NoOfDigitsInURL': kwargs.get('NoOfDigitsInURL', 0),
        'DigitRatioInURL': kwargs.get('DigitRatioInURL', 0),
        'NoOfSpecialCharsInURL': kwargs.get('NoOfSpecialCharsInURL', 0),
        'SpacialCharRatioInURL': kwargs.get('SpacialCharRatioInURL', 0),
        'IsHTTPS': kwargs.get('IsHTTPS', 1),
        'LineOfCode': kwargs.get('LineOfCode', 0),
        'LargestLineLength': kwargs.get('LargestLineLength', 0),
        'DomainTitleMatchScore': kwargs.get('DomainTitleMatchScore', 0),
        'URLTitleMatchScore': kwargs.get('URLTitleMatchScore', 0),
        'NoOfURLRedirect': kwargs.get('NoOfURLRedirect', 0),
        'NoOfSelfRedirect': kwargs.get('NoOfSelfRedirect', 0),
        'HasDescription': kwargs.get('HasDescription', 1),
        'HasExternalFormSubmit': kwargs.get('HasExternalFormSubmit', 0),
        'HasSocialNet': kwargs.get('HasSocialNet', 0),
        'HasSubmitButton': kwargs.get('HasSubmitButton', 1),
        'HasHiddenFields': kwargs.get('HasHiddenFields', 0),
        'HasPasswordField': kwargs.get('HasPasswordField', 0),
        'NoOfImage': kwargs.get('NoOfImage', 0),
        'NoOfCSS': kwargs.get('NoOfCSS', 0),
        'NoOfJS': kwargs.get('NoOfJS', 0),
        'NoOfSelfRef': kwargs.get('NoOfSelfRef', 0),
        'NoOfEmptyRef': kwargs.get('NoOfEmptyRef', 0),
        'NoOfExternalRef': kwargs.get('NoOfExternalRef', 0),
    }
    
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)[0]
    
    if prediction == 1:
        print("Suspicious")
    else:
        print("Genuine")

if __name__ == "__main__":
    # Update the path to your CSV file
    file_path = '/mnt/data/phishing_dataset.csv'  # Replace with the actual file path
    X, y = load_and_prepare_data(file_path)
    
    # Train the model
    model = train_model(X, y)
    
    # Example URL classification (adjust inputs accordingly)
    user_url = "https://example.com"
    classify_url(model, user_url, DomainLength=15, URLSimilarityIndex=100, CharContinuationRate=0.9, TLDLegitimateProb=0.8)

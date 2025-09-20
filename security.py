import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Step 1: Define Your Data Directories ---
VULNERABLE_BASE_DIR = 'smartbugs-curated-main/dataset'
SECURE_DIR = 'secure_contracts'


# --- Step 2: Read and Label the Data (with Debugging) ---
def load_and_label_contracts():
    """
    Loads all contract code from the directories and labels it.
    Label 1 = Vulnerable, Label 0 = Secure.
    """
    print("--- Starting to load contracts ---") # DEBUG
    contracts = []
    labels = []

    # --- This is the part we need to debug ---
    # Load all secure contracts and label them as 0
    print(f"\n[1] Checking for SECURE contracts in: {SECURE_DIR}") # DEBUG
    if os.path.exists(SECURE_DIR):
        print(f"--> SUCCESS: Directory '{SECURE_DIR}' exists.") # DEBUG
        try:
            file_list = os.listdir(SECURE_DIR)
            print(f"--> Files found in directory: {file_list}") # DEBUG
            
            if not file_list:
                print("--> WARNING: The directory is empty.") # DEBUG

            for filename in file_list:
                if filename.endswith(".sol"):
                    print(f"    - Processing secure file: {filename}") # DEBUG
                    file_path = os.path.join(SECURE_DIR, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        contracts.append(file.read())
                        labels.append(0)
                else:
                    print(f"    - Skipping non-Solidity file: {filename}") # DEBUG
        except Exception as e:
            print(f"--> ERROR: Could not read directory or files. Reason: {e}") # DEBUG
    else:
        print(f"--> CRITICAL FAILURE: The directory '{SECURE_DIR}' was not found.") # DEBUG
    
    # Load all vulnerable contracts and label them as 1
    print(f"\n[2] Checking for VULNERABLE contracts in: {VULNERABLE_BASE_DIR}") # DEBUG
    for root, dirs, files in os.walk(VULNERABLE_BASE_DIR):
        for filename in files:
            if filename.endswith(".sol"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    contracts.append(file.read())
                    labels.append(1)

    # --- Final Count ---
    print("\n[3] Final counts before returning:") # DEBUG
    print(f"--> Total contracts loaded: {len(contracts)}") # DEBUG
    print(f"--> Total labels loaded: {len(labels)}") # DEBUG
    if labels:
        print(f"--> Count of 'secure' labels (0): {labels.count(0)}") # DEBUG
        print(f"--> Count of 'vulnerable' labels (1): {labels.count(1)}") # DEBUG

    return contracts, labels


# --- Step 3: Prepare Data for the AI Model ---
def train_and_save_model(contracts, labels):
    """
    Performs k-fold cross-validation, trains a final model, and saves it.
    """
    print("\n--- Starting model training ---")
    contracts = np.array(contracts)
    labels = np.array(labels)
    
    if len(np.unique(labels)) < 2:
        print("Error: The dataset must contain at least two classes (vulnerable and secure) to train a model.")
        return

    # Use K-Fold Cross-Validation for robust model evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    model = LogisticRegression(max_iter=1000)

    for train_index, test_index in kf.split(contracts):
        X_train, X_test = contracts[train_index], contracts[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        
        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_test_vectorized)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    print(f"K-Fold Cross-Validation Accuracy Scores: {accuracy_scores}")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")
    
    # --- Step 4: Train the Final Model on the Full Dataset ---
    full_corpus_vectorized = vectorizer.fit_transform(contracts)
    model.fit(full_corpus_vectorized, labels)
    
    # --- Step 5: Save the Trained Model and Vectorizer ---
    with open('security_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('security_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\nFinal Model trained on full dataset and saved as security_model.pkl and security_vectorizer.pkl")

# --- THIS IS THE CRUCIAL PART THAT WAS LIKELY MISSING ---
if __name__ == "__main__":
    print("\n--- Script execution started ---")
    contracts, labels = load_and_label_contracts()
    
    if contracts and len(np.unique(labels)) >= 2:
        train_and_save_model(contracts, labels)
    else:
        print("\n--- Halting execution: Not enough data or classes to train a model. ---")
        if not contracts:
            print("Reason: No Solidity files were found at all.")
        else:
            print("Reason: Could not find contracts for both 'secure' and 'vulnerable' classes.")
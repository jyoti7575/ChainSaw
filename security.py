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


# --- Step 2: Read and Label the Data ---
def load_and_label_contracts():
    """
    Loads all contract code from the directories and labels it.
    Label 1 = Vulnerable, Label 0 = Secure.
    """
    print("--- Loading smart contracts from directories... ---")
    contracts = []
    labels = []

    # Load secure contracts and label them as 0
    if os.path.exists(SECURE_DIR):
        for filename in os.listdir(SECURE_DIR):
            if filename.endswith(".sol"):
                file_path = os.path.join(SECURE_DIR, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    contracts.append(file.read())
                    labels.append(0)

    # Load vulnerable contracts and label them as 1
    for root, dirs, files in os.walk(VULNERABLE_BASE_DIR):
        for filename in files:
            if filename.endswith(".sol"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    contracts.append(file.read())
                    labels.append(1)

    print(f"--> Found {labels.count(0)} secure contracts.")
    print(f"--> Found {labels.count(1)} vulnerable contracts.")
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
        print("Error: The dataset must contain at least two classes (vulnerable and secure).")
        return

    # Use K-Fold Cross-Validation for robust model evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    
    # This is the key change to handle imbalanced data
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    print("--- Performing 5-Fold Cross-Validation ---")
    for fold, (train_index, test_index) in enumerate(kf.split(contracts), 1):
        X_train, X_test = contracts[train_index], contracts[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        
        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_test_vectorized)
        score = accuracy_score(y_test, y_pred)
        accuracy_scores.append(score)
        print(f"Fold {fold} Accuracy: {score:.2f}")

    print(f"\n--> Mean Cross-Validation Accuracy: {np.mean(accuracy_scores):.2f}")
    
    # --- Step 4: Train the Final Model on the Full Dataset ---
    print("\n--- Training final model on the entire dataset... ---")
    full_corpus_vectorized = vectorizer.fit_transform(contracts)
    model.fit(full_corpus_vectorized, labels)
    
    # --- Step 5: Save the Trained Model and Vectorizer ---
    with open('security_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('security_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\n✅ Final model trained and saved successfully!")


if __name__ == "__main__":
    contracts, labels = load_and_label_contracts()
    
    if contracts and len(np.unique(labels)) >= 2:
        train_and_save_model(contracts, labels)
    else:
        print("\nExecution halted: Not enough data or classes to train a model.")
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os

# Import different models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Paths for model and dataset
DATA_PATH = os.path.join('ai_model', 'training_data.csv')

# =======================
# Train the Model with Cross-Validation
# =======================
def train_model_with_cross_validation():
    """
    Train different machine learning models using financial transaction data
    and display cross-validation results.
    """
    # Check if the dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    # Load training data
    data = pd.read_csv(DATA_PATH)

    # Add target column: binary classification for savings behavior
    data['target'] = (data['Disposable_Income'] >= data['Desired_Savings']).astype(int)

    # Select features and target
    features = data[[
        'Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
        'Healthcare', 'Education', 'Miscellaneous'
    ]]
    target = data['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # List of models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Support Vector Machine': SVC(class_weight='balanced', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
    }

    # Store cross-validation results
    cv_results = []

    # Iterate through models, train and evaluate each with cross-validation
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} with Cross-Validation...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        
        # Calculate mean cross-validation score
        mean_cv_score = cv_scores.mean()
        cv_results.append({'Model': model_name, 'Cross-Validation Accuracy': mean_cv_score})

        print(f"{model_name} Cross-Validation Accuracy: {mean_cv_score}")
        print(f"Cross-Validation Scores: {cv_scores}")
        
        # Save the trained model (optional)
        model_file_path = os.path.join('ai_model', f'{model_name.lower().replace(" ", "_")}_model.pkl')
        with open(model_file_path, "wb") as file:
            pickle.dump(model, file)
        print(f"{model_name} saved to {model_file_path}")
    
    # Display the cross-validation results in a DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    print("\nCross-Validation Accuracy of All Models:")
    print(cv_results_df)


# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    # Train the models and evaluate them using cross-validation
    print("Training and evaluating different models with cross-validation...")
    train_model_with_cross_validation()

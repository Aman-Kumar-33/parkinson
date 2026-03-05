# This script demonstrates how to generate the values for a confusion matrix.
# We will use the scikit-learn library, a standard tool for machine learning in Python.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification

# --- Step 1: Create a Simulated Dataset ---
# We need data to work with. `make_classification` creates a sample dataset
# with features (X) and corresponding labels (y).
# Let's imagine we're predicting if a patient has a disease (1) or not (0).
X, y = make_classification(
    n_samples=165,    # Total number of "patients" in our data
    n_features=10,    # Number of medical measurements for each patient
    n_informative=5,  # Number of measurements that are actually useful
    n_redundant=0,
    n_classes=2,      # Two outcomes: 0 (No Disease) and 1 (Disease)
    weights=[0.6, 0.4],# 60% are healthy, 40% have the disease
    flip_y=0.05,      # Introduce some noise/difficulty
    random_state=42   # Ensures the same "random" data is generated every time
)

# --- Step 2: Split Data into Training and Testing Sets ---
# We train the model on one part of the data (training set) and test it
# on another, unseen part (testing set) to see how well it performs.
# y_test contains the TRUE answers we will compare against.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Step 3: Create and Train a Machine Learning Model ---
# We'll use Logistic Regression, a common and simple classification model.
# The .fit() method is where the model "learns" from the training data.
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Step 4: Make Predictions on the Test Data ---
# Now we ask the trained model to predict the outcomes for the test set.
# y_pred will contain the model's GUESSES.
y_pred = model.predict(X_test)

# --- Step 5: Generate the Confusion Matrix ---
# This is the key step. We compare the true answers (y_test) with
# the model's guesses (y_pred).
# scikit-learn's confusion_matrix function does this for us.
# Note: In scikit-learn, the convention is:
#       C[0,0] = True Negative
#       C[0,1] = False Positive
#       C[1,0] = False Negative
#       C[1,1] = True Positive
# (Assuming class 0 is Negative and class 1 is Positive)
cm = confusion_matrix(y_test, y_pred)

print("--- Confusion Matrix Results ---")
print("The generated confusion matrix is a 2x2 NumPy array:")
print(cm)
print("\n------------------------------------")


# --- Step 6: Extract and Label Each Value ---
# To make it crystal clear, let's pull out each value from the matrix.
# .ravel() flattens the 2x2 matrix into a simple 1D array [TN, FP, FN, TP]
try:
    tn, fp, fn, tp = cm.ravel()
    print("These are the numbers you need for your HTML visualization:\n")
    print(f"True Positives (TP): {tp}")
    print(f"-> Correctly predicted 'Positive' (e.g., has disease)")
    print("-" * 20)
    print(f"False Negatives (FN): {fn}")
    print(f"-> Incorrectly predicted 'Negative' (e.g., has disease but model said no)")
    print("-" * 20)
    print(f"False Positives (FP): {fp}")
    print(f"-> Incorrectly predicted 'Positive' (e.g., healthy but model said has disease)")
    print("-" * 20)
    print(f"True Negatives (TN): {tn}")
    print(f"-> Correctly predicted 'Negative' (e.g., healthy)")
    print("-" * 20)

except ValueError:
    print("Could not unpack the confusion matrix. It might not be 2x2.")
    print("This can happen if your test data only contains one class.")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_accuracy_boxplot(results_df, save_path='model_accuracy_comparison.png'):
    """
    Generates and saves a boxplot to compare model accuracies from real cross-validation results.
    """
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=results_df)
    sns.stripplot(data=results_df, jitter=True, color='black', size=5, alpha=0.5)

    plt.title('Model Accuracy Distribution (from 10-Fold Cross-Validation)', fontsize=16, weight='bold')
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.xlabel('Machine Learning Model', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"\n[SUCCESS] Boxplot visualization saved to {save_path}")
    print("This plot is generated from the real cross-validation results.")


def run_real_data_analysis():
    """
    Performs a full statistical validation using the provided CSV data,
    and generates a plot from the real results.
    """
    # 1. Load Real Data
    try:
        df = pd.read_csv('parkinsons_hospital.csv')
        print("Successfully loaded 'parkinsons_hospital.csv'.")
    except FileNotFoundError:
        print("[ERROR] 'parkinsons_hospital.csv' not found. Please ensure it's in the same directory.")
        return

    # 2. Preprocess Data
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
    
    X = df.drop('status', axis=1)
    y = df['status']

    # 3. Define Models
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC(probability=True, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('mlp', MLPClassifier(max_iter=1000, random_state=42))
    ]

    stacking_classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5 # Inner CV for the stacking model
    )
    
    models = {
        "Logistic Regression": estimators[0][1],
        "KNN": estimators[1][1],
        "SVC": estimators[2][1],
        "Random Forest": estimators[3][1],
        "Gradient Boosting": estimators[4][1],
        "MLP": estimators[5][1],
        "Stacking Classifier": stacking_classifier
    }

    # 4. Perform Stratified K-Fold Cross-Validation (Outer CV)
    N_SPLITS = 10
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    cv_results = {name: [] for name in models.keys()}

    print("\nStarting 10-fold cross-validation...")
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"--- Processing Fold {fold+1}/{N_SPLITS} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            cv_results[name].append(accuracy)
    print("Cross-validation complete.")

    # 5. Analyze and Display Statistical Results
    cv_results_df = pd.DataFrame(cv_results)
    
    print("\n" + "="*60)
    print("      STATISTICAL ANALYSIS OF MODEL ACCURACY (REAL DATA)")
    print("="*60 + "\n")

    print("--- Descriptive Statistics from 10-Fold Cross-Validation ---\n")
    summary_stats = cv_results_df.agg(['mean', 'std', 'var']).T
    summary_stats.columns = ['Mean Accuracy', 'Standard Deviation', 'Variance']
    print(summary_stats)
    print("\n" + "-"*60 + "\n")

    print("--- Paired t-test: Stacking Classifier vs. Base Models ---\n")
    stacking_scores = cv_results_df['Stacking Classifier']
    
    for name in models.keys():
        if name != 'Stacking Classifier':
            model_scores = cv_results_df[name]
            t_statistic, p_value = stats.ttest_rel(stacking_scores, model_scores)
            
            print(f"Comparing Stacking Classifier with {name}:")
            print(f"  t-statistic: {t_statistic:.4f}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  Result: The difference is STATISTICALLY SIGNIFICANT. ✅\n")
            else:
                print("  Result: The difference is NOT statistically significant. ❌\n")

    # 6. Generate and save the boxplot visualization from the real results
    create_accuracy_boxplot(cv_results_df)

if __name__ == "__main__":
    run_real_data_analysis()
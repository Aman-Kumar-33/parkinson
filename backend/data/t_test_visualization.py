import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# --- Parameters from your t-test results ---
# Comparing Stacking Classifier vs. Logistic Regression
t_statistic = 10.8281
degrees_of_freedom = 9  # For a paired test, df = n - 1 = 10 - 1 = 9
alpha = 0.05            # Significance level

# --- Generate data for the t-distribution curve ---
x = np.linspace(-5, 12, 1000)
y = t.pdf(x, df=degrees_of_freedom)

# --- Calculate the critical value ---
# We use a one-tailed test because we are testing if the Stacking model is *better*
critical_value = t.ppf(1 - alpha, df=degrees_of_freedom)

# --- Create the plot ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the t-distribution
ax.plot(x, y, 'b-', label=f't-distribution (df={degrees_of_freedom})')

# Shade the rejection region
rejection_x = np.linspace(critical_value, 12, 100)
rejection_y = t.pdf(rejection_x, df=degrees_of_freedom)
ax.fill_between(rejection_x, rejection_y, color='red', alpha=0.5, label=f'Rejection Region (α={alpha})')

# Mark the critical value
ax.axvline(critical_value, color='red', linestyle='--', linewidth=2, label=f'Critical Value = {critical_value:.2f}')

# Mark the t-statistic
ax.axvline(t_statistic, color='green', linestyle='-', linewidth=2.5, label=f'Observed t-statistic = {t_statistic:.2f}')

# --- Add labels and title ---
ax.set_title('Hypothesis Test Visualization: Stacking vs. Logistic Regression', fontsize=16, weight='bold')
ax.set_xlabel('t-value', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(bottom=0)
ax.set_xlim(left=-4)

# Add text to explain the result
ax.text(t_statistic + 0.3, 0.1, 'Reject H₀', color='green', weight='bold', fontsize=12)
ax.text(critical_value - 2.5, 0.1, 'Fail to Reject H₀', color='black', fontsize=12)

plt.tight_layout()
plt.savefig('t_test_visualization.png', dpi=300)
print("Plot saved to t_test_visualization.png")
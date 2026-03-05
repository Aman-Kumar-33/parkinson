import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# Create a directory to save the images
if not os.path.exists('image'):
    os.makedirs('image')

# Load the datasets
try:
    train_df = pd.read_csv('parkinsons_hospital.csv')
    test_df = pd.read_csv('Parkinsson_data.csv')
except FileNotFoundError:
    print("Make sure 'parkinsons_hospital.csv' and 'Parkinsson_data.csv' are in the correct directory.")
    exit()

# --- Heatmap of the training data ---
plt.figure(figsize=(20, 15))
corr = train_df.drop('name', axis=1).corr()
sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm')
plt.title('Correlation Heatmap of Training Data')
plt.savefig('image/heatmap.png', bbox_inches='tight')
plt.close()
print("Generated heatmap and saved to image/heatmap.png")

# --- Top 10 Feature Distribution Plots ---
train_features = train_df.drop('name', axis=1)
corr_with_status = train_features.corr()['status'].abs().sort_values(ascending=False)
top_10_features = corr_with_status.drop('status').head(10).index.tolist()

print("Top 10 most correlated features with 'status':")
print(top_10_features)

# Create and save distribution plots for the top 10 features
for feature in top_10_features:
    # Sanitize the feature name to create a valid filename
    sanitized_feature_name = re.sub(r'[^\w\.-]', '_', feature)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x=feature, hue='status', kde=True, element='step')
    plt.title(f'Distribution of {feature} by Parkinson\'s Status')
    
    image_path = f'image/{sanitized_feature_name}_distribution.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    print(f"Generated distribution plot for {feature} and saved to {image_path}")

print("\nImage generation complete. Check the 'image' folder.")
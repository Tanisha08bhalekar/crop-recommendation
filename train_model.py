import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Example: Load dataset (replace with your dataset file name)
data = pd.read_csv("crop_data.csv")  # Must be in the same folder

# Features and target
X = data.drop('label', axis=1)
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ model.pkl created successfully!")

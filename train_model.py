import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load your dataset
data = pd.read_csv("phishing.csv")

# Assume the last column is the label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train the model
model = GradientBoostingClassifier()
model.fit(X, y)

# Save it
with open("pickle/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved to pickle/model.pkl")

import pandas as pd
import joblib

# ==== Load dataset ====
data_path = "your_dataset.csv"   # Change this to your dataset file
df = pd.read_csv(data_path)

# ==== Preprocess (match same preprocessing as training) ====
# Example: drop CustomerID, encode categorical, scale numerical
X = df.drop(columns=["customerID"])  

# ==== Load trained model ====
model = joblib.load("trained_model.pkl")  # Change path if needed

# ==== Run predictions ====
predictions = model.predict(X)

# ==== Save results ====
df["Predicted_Churn"] = predictions
df.to_csv("predictions.csv", index=False)

print("✅ Predictions saved to predictions.csv")


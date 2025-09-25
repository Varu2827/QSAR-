import pandas as pd
import joblib

# Load external data
ext_df = pd.read_csv("external_compounds.csv")

# Load trained QSAR model
model = joblib.load("qsar_rf_model.joblib")

# Predict probabilities and labels
probabilities = model.predict_proba(ext_df)[:, 1]  # probability of class 1 (Activator)
predictions = model.predict(ext_df)

# Add to dataframe
ext_df["Predicted_Activity"] = predictions
ext_df["Confidence"] = probabilities

# Save the output
ext_df.to_csv("predicted_activators.csv", index=False)
print("✅ Predictions saved to predicted_activators.csv")

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(ext_df["Confidence"], bins=10, kde=True)
plt.title("Prediction Confidence for External Compounds")
plt.xlabel("Probability of Being Activator")
plt.ylabel("Number of Compounds")
plt.show()

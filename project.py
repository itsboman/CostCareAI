import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import streamlit as st

# 🩺 Load your dataset
df = pd.read_csv("healthcare.csv")  # <- change this to your file name
# Drop unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# (Optional) drop derived features if they exist
df = df.drop(columns=['age_group', 'bmi_category'], errors='ignore')

le = LabelEncoder()

# 🔤 Encode categorical columns automatically
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()

for col in cat_cols:
    print(f"Encoding column: {col}")
    df[col] = le.fit_transform(df[col].astype(str))

# 🧩 Split features & target
X = df.drop("charges", axis=1)  # replace 'charges' with your target column name if different
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 📈 Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model trained successfully!")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")












import gspread
from google.oauth2.service_account import Credentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
client = gspread.authorize(creds)

sheet = client.open("health").sheet1
st.success("✅ Connected to Google Sheets successfully!")

# Test reading the first few rows
data = sheet.get_all_records()
st.write("Here’s your Google Sheet data preview 👇")
st.dataframe(data[:5])






# 💾 Save model using joblib
joblib.dump(model, "model.pkl")
print("💾 Model saved as model.pkl")
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# Define scope for Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load your service account credentials (JSON file)
creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
client = gspread.authorize(creds)

# Open your sheet by name or URL
sheet = client.open("health").sheet1

# Convert to DataFrame
data = pd.DataFrame(sheet.get_all_records())

st.write("📊 Live Hospital Data:")
st.dataframe(data)


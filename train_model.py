import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")
df = pd.read_csv('hotel_bookings.csv')

print("Applying Feature Engineering...")
# 1. Feature Engineering: Create meaningful hospitality metrics
df['total_guests'] = df['adults'] + df['children']
df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['is_family'] = np.where(df['children'] > 0, 1, 0)

# Filter out invalid bookings (0 guests or 0 nights)
df = df[(df['total_guests'] > 0) & (df['total_stay_nights'] > 0)].copy()

# 2. Encode Categorical Features for ML
cat_cols = ['hotel', 'arrival_date_month', 'market_segment', 'deposit_type', 'customer_type']
label_encoders = {}

df_ml = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    label_encoders[col] = le

# Define Features (X) and Target (y)
X = df_ml.drop(columns=['is_canceled'])
y = df_ml['is_canceled']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Random Forest Classifier model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
clf.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Extract Feature Importances (Top Cancellation Indicators)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("--- Top Early Cancellation Indicators ---")
print(feature_importances.head(6).to_string(index=False))
print("------------------------------------------\n")

# Save cleaned dataset for Power BI visualization
df.to_csv('powerbi_hospitality_data.csv', index=False)
print("Saved processed data to 'powerbi_hospitality_data.csv'.")
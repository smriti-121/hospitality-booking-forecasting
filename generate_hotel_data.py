import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 32000

print(f"Generating synthetic dataset with {n_samples} hotel reservations...")

# 1. Core Feature Generation
hotel_types = np.random.choice(['City Hotel', 'Resort Hotel'], size=n_samples, p=[0.65, 0.35])
lead_times = np.random.exponential(scale=100, size=n_samples).astype(int)
lead_times = np.clip(lead_times, 0, 700)

months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']
month_weights = [0.06, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.13, 0.09, 0.08, 0.06, 0.06]
arrival_months = np.random.choice(months, size=n_samples, p=month_weights)

stays_weekend = np.random.poisson(lam=1, size=n_samples)
stays_week = np.random.poisson(lam=2.5, size=n_samples)

adults = np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.75, 0.05])
children = np.random.choice([0, 1, 2], size=n_samples, p=[0.85, 0.10, 0.05])

market_segments = np.random.choice(
    ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups'], 
    size=n_samples, 
    p=[0.45, 0.20, 0.15, 0.10, 0.10]
)

deposit_types = np.random.choice(
    ['No Deposit', 'Non Refund', 'Refundable'], 
    size=n_samples, 
    p=[0.85, 0.14, 0.01]
)

adr = np.round(np.random.normal(loc=105, scale=35, size=n_samples), 2)
adr = np.clip(adr, 20.0, 350.0)

previous_cancellations = np.random.choice([0, 1, 2], size=n_samples, p=[0.90, 0.08, 0.02])
special_requests = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.60, 0.25, 0.10, 0.05])

# 2. Cancellation Logic (Logistic Probability Function)
# Longer lead time, no special requests, and previous cancellations increase cancellation probability
logits = (
    -1.2 
    + 0.005 * lead_times 
    + 1.8 * (deposit_types == 'Non Refund') 
    + 0.5 * previous_cancellations 
    - 0.4 * special_requests 
    + 0.002 * adr
)
probabilities = 1 / (1 + np.exp(-logits))
is_canceled = (np.random.binomial(1, probabilities)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'hotel': hotel_types,
    'is_canceled': is_canceled,
    'lead_time': lead_times,
    'arrival_date_month': arrival_months,
    'stays_in_weekend_nights': stays_weekend,
    'stays_in_week_nights': stays_week,
    'adults': adults,
    'children': children,
    'market_segment': market_segments,
    'deposit_type': deposit_types,
    'customer_type': np.random.choice(['Transient', 'Transient-Party', 'Contract', 'Group'], size=n_samples, p=[0.75, 0.18, 0.05, 0.02]),
    'adr': adr,
    'previous_cancellations': previous_cancellations,
    'total_of_special_requests': special_requests
})

df.to_csv('hotel_bookings.csv', index=False)
print(f"Success! Saved {len(df)} rows to 'hotel_bookings.csv'.")
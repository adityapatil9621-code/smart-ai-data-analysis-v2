import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

# Features
age = np.random.randint(18, 60, n)
income = np.random.randint(20000, 120000, n)
time_on_app = np.random.uniform(5, 60, n)
discount_used = np.random.uniform(0, 50, n)

# Categorical
payment_method = np.random.choice(["Card", "UPI", "Wallet"], n)
subscription = np.random.choice(["Basic", "Premium"], n)

# Target (strong relationship)
spending_score = (
    0.3 * income +
    5 * time_on_app +
    2 * discount_used -
    1.5 * age +
    np.random.normal(0, 1000, n)  # small noise
)

df = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Time_On_App": time_on_app,
    "Discount_Used": discount_used,
    "Payment_Method": payment_method,
    "Subscription": subscription,
    "Spending_Score": spending_score
})

df.to_csv("smart_ai_dataset.csv", index=False)

print("Dataset generated successfully!")
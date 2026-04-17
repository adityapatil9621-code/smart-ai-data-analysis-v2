import pandas as pd
import numpy as np

np.random.seed(42)

rows = 500

# Time index
time_index = np.arange(rows)

# Marketing Spend (main driver)
marketing_spend = np.random.normal(5000, 800, rows)

# Revenue (strongly correlated with marketing)
monthly_spend = marketing_spend * 2.5 + np.random.normal(0, 2000, rows)

# Add slight downward trend
monthly_spend = monthly_spend - (time_index * 5)

# Add anomalies (~6%)
anomaly_indices = np.random.choice(rows, size=int(rows * 0.06), replace=False)
monthly_spend[anomaly_indices] *= np.random.uniform(1.8, 2.5, len(anomaly_indices))

# Engagement metric correlated to revenue
session_duration = monthly_spend / 100 + np.random.normal(0, 10, rows)

# High volatility cost metric
discount_rate = np.random.normal(20, 15, rows)

# Categorical segmentation
segments = np.random.choice(
    ["Premium", "Standard", "Basic"],
    rows,
    p=[0.3, 0.5, 0.2]
)

# Adjust revenue by segment
monthly_spend += np.where(segments == "Premium", 3000, 0)
monthly_spend -= np.where(segments == "Basic", 2000, 0)

# Region category
regions = np.random.choice(
    ["North", "South", "West", "East"],
    rows
)

# Redundant feature (correlation > 0.9)
redundant_copy = monthly_spend * 1.01 + np.random.normal(0, 50, rows)

# Create DataFrame
df = pd.DataFrame({
    "TimeIndex": time_index,
    "MarketingSpend": marketing_spend,
    "MonthlySpend": monthly_spend,
    "SessionDurationMinutes": session_duration,
    "DiscountRate": discount_rate,
    "CustomerSegment": segments,
    "Region": regions,
    "RevenueMirror": redundant_copy
})

df.to_csv("superior_analysis_test_dataset.csv", index=False)

print("Dataset generated successfully!")

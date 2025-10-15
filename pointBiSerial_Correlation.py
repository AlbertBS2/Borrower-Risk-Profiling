from scipy.stats import pointbiserialr
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data

loan_data = "data/accepted_2007_to_2018Q4.csv.gz"
unemployment_rate_data = ["data/unemployment_rate_0.csv", "data/unemployment_rate_1.csv", "data/unemployment_rate_2.csv", "data/unemployment_rate_3.csv", "data/unemployment_rate_4.csv"]

data = preprocess_data(loan_data, unemployment_rate_data)
target = data['default']
features = data.drop(columns=['default'])
numeric_cols = features.select_dtypes(include=[np.number]).columns

correlation_scores = []

for col in numeric_cols:
    try:
        series = features[col].fillna(features[col].median())
        r, _ = pointbiserialr(target, series)
        correlation_scores.append((col, abs(r)))
    except Exception as e:
        print(f"Skipped {col} due to error: {e}")

corr_df = pd.DataFrame(correlation_scores, columns=['feature', 'abs_pointbiserial_corr'])
corr_df = corr_df.sort_values('abs_pointbiserial_corr', ascending=False)

print("\nTop 20 Features by Point-Biserial Correlation:")
print(corr_df.head(20))

# Save
corr_df.to_csv("pointbiserial_correlation_scores.csv", index=False)

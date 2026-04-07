import pandas as pd

# Load dataset
df = pd.read_csv("Attempt_1/digit.csv")

label_col = "label"

num_files = 5
samples_per_file = 500

# Get class proportions
class_counts = df[label_col].value_counts(normalize=True)

for i in range(num_files):
    sampled_parts = []

    for cls, proportion in class_counts.items():
        n_samples = int(proportion * samples_per_file)  # avoid over-rounding
        
        cls_samples = df[df[label_col] == cls].sample(
            n=n_samples,
            replace=True,
            random_state=42 + i
        )
        sampled_parts.append(cls_samples)

    combined = pd.concat(sampled_parts)

    # 🔥 FIX: allow replacement if needed
    final_df = combined.sample(
        n=samples_per_file,
        replace=(len(combined) < samples_per_file),
        random_state=42 + i
    )

    filename = f"digit_split_{i+1}.csv"
    final_df.to_csv(filename, index=False)

    print(f"Saved {filename} with {len(final_df)} samples")
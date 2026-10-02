import pandas as pd
import ast
import numpy as np

rows = []

for driver in range(1, 6):
    file_path = f"data/train/D{driver}traindata50.csv"
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        window = ast.literal_eval(row["data"])

        samples = np.array([
            [float(value) for value in sample.split()]
            for sample in window
        ])

     
        z_raw = samples[:, 2]

       
        z_kf = samples[:, 5]

        rows.append({
            "driver": driver,
            "label": row["label"],
            "peak_abs_z_raw": np.max(np.abs(z_raw)),
            "peak_abs_z_kf": np.max(np.abs(z_kf))
        })

results = pd.DataFrame(rows)

print("LONGITUDINAL ACCELERATION COMPARISON")
print("------------------------------------")

for label in ["NORMAL", "DROWSY", "AGGRESSIVE"]:
    subset = results[results["label"] == label]

    print(f"\n{label}")
    print(f"Windows: {len(subset)}")

    print(
        "Raw Z median peak: "
        f"{subset['peak_abs_z_raw'].median():.4f} G"
    )

    print(
        "Z_KF median peak:  "
        f"{subset['peak_abs_z_kf'].median():.4f} G"
    )
    
print("\nZ_KF PEAK DISTRIBUTION")
print("----------------------")

for label in ["NORMAL", "DROWSY", "AGGRESSIVE"]:
    subset = results[results["label"] == label]["peak_abs_z_kf"]

    print(f"\n{label}")
    print(f"25th percentile: {subset.quantile(0.25):.4f} G")
    print(f"Median:          {subset.quantile(0.50):.4f} G")
    print(f"75th percentile: {subset.quantile(0.75):.4f} G")
    print(f"90th percentile: {subset.quantile(0.90):.4f} G")
    print(f"95th percentile: {subset.quantile(0.95):.4f} G")
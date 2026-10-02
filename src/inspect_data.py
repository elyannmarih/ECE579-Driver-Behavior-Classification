import pandas as pd

all_drivers = []

for driver in range(1, 6):
    file_path = f"data/train/D{driver}traindata50.csv"
    df = pd.read_csv(file_path)

    # Keep track of which driver each window came from
    df["driver"] = driver
    all_drivers.append(df)

    print(f"\nDriver {driver}")
    print("Windows:", len(df))
    print(df["label"].value_counts())

# Combine Drivers 1-5
combined_df = pd.concat(all_drivers, ignore_index=True)

print("\n-----------------------------")
print("DRIVERS 1-5 COMBINED")
print("-----------------------------")
print("Total windows:", len(combined_df))

print("\nClass counts:")
print(combined_df["label"].value_counts())

print("\nClass percentages:")
print(combined_df["label"].value_counts(normalize=True) * 100)
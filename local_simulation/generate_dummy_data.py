import os
import numpy as np
import pandas as pd

# The 40 features as defined in the data dictionary
FEATURE_COLUMNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]


def generate_data():
    os.makedirs("local_simulation/dummy_data", exist_ok=True)

    for i in range(1, 6):
        n_samples = np.random.randint(500, 1000)

        # Generate random floats for features
        X = np.random.randn(n_samples, 40) * 10

        # Generate binary target (imbalanced to mimic Sepsis)
        y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        df["SepsisLabel"] = y

        # Introduce some fake NaNs to test preprocessing
        df.loc[::10, "HR"] = np.nan

        file_path = f"local_simulation/dummy_data/hospital_{i}.parquet"
        df.to_parquet(file_path)
        print(f"Generated {file_path} with {n_samples} rows.")


if __name__ == "__main__":
    generate_data()

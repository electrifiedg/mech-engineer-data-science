from pathlib import Path
import pandas as pd
import numpy as np
import joblib

CLEAN_PATH = Path(__file__).parent.parent / "data" / "compressor_clean.parquet"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
RESULTS_PATH = OUTPUTS_DIR / "flagged_results.csv"


def main():
    print("=== Stage 3: Flag Deviations ===")

    df = pd.read_parquet(CLEAN_PATH)
    print(f"  Loaded {len(df)} rows")

    poly = joblib.load(OUTPUTS_DIR / "poly_features.pkl")
    model_capacity = joblib.load(OUTPUTS_DIR / "model_capacity.pkl")
    model_power = joblib.load(OUTPUTS_DIR / "model_power.pkl")

    X = df[["T_e", "T_c"]].values
    Phi = poly.transform(X)

    df["capacity_pred"] = model_capacity.predict(Phi)
    df["power_pred"] = model_power.predict(Phi)
    df["capacity_residual"] = df["Capacity"] - df["capacity_pred"]
    df["power_residual"] = df["Power"] - df["power_pred"]

    cap_thresh = 2 * df["capacity_residual"].std()
    pow_thresh = 2 * df["power_residual"].std()
    print(f"  Capacity threshold (±2σ): ±{cap_thresh:.2f}")
    print(f"  Power threshold    (±2σ): ±{pow_thresh:.2f}")

    df["capacity_flag"] = df["capacity_residual"].abs() > cap_thresh
    df["power_flag"] = df["power_residual"].abs() > pow_thresh
    df["any_flag"] = df["capacity_flag"] | df["power_flag"]

    n_cap = df["capacity_flag"].sum()
    n_pow = df["power_flag"].sum()
    n_any = df["any_flag"].sum()

    print(f"\n  Total rows:              {len(df)}")
    print(f"  Flagged on Capacity:     {n_cap}")
    print(f"  Flagged on Power:        {n_pow}")
    print(f"  Flagged on either:       {n_any}")

    if n_any > 0:
        print("\n  Flagged rows:")
        flagged = df[df["any_flag"]][
            ["T_e", "T_c", "capacity_residual", "power_residual",
             "capacity_flag", "power_flag"]
        ]
        print(flagged.to_string(index=True))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Results written to {RESULTS_PATH}")
    print("=== Flag complete ===\n")


if __name__ == "__main__":
    main()

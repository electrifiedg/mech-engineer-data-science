from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

CLEAN_PATH = Path(__file__).parent.parent / "data" / "compressor_clean.parquet"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def main():
    print("=== Stage 2: Fit Curve ===")

    df = pd.read_parquet(CLEAN_PATH)
    print(f"  Loaded {len(df)} rows from {CLEAN_PATH.name}")

    X = df[["T_e", "T_c"]].values

    poly = PolynomialFeatures(degree=3, include_bias=True)
    Phi = poly.fit_transform(X)
    print(f"  Polynomial features shape: {Phi.shape} ({Phi.shape[1]} terms)")

    model_capacity = LinearRegression().fit(Phi, df["Capacity"].values)
    model_power = LinearRegression().fit(Phi, df["Power"].values)

    r2_cap = model_capacity.score(Phi, df["Capacity"].values)
    r2_pow = model_power.score(Phi, df["Power"].values)
    print(f"  R² Capacity: {r2_cap:.4f}")
    print(f"  R² Power:    {r2_pow:.4f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(poly, OUTPUTS_DIR / "poly_features.pkl")
    joblib.dump(model_capacity, OUTPUTS_DIR / "model_capacity.pkl")
    joblib.dump(model_power, OUTPUTS_DIR / "model_power.pkl")
    print(f"  Serialized models to {OUTPUTS_DIR}")
    print("=== Fit complete ===\n")


if __name__ == "__main__":
    main()

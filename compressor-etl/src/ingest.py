from pathlib import Path
import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/PurdueMechanicalEngineering/"
    "me-239-intro-to-data-science/master/data/compressor_data.xlsx"
)
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "compressor_data.xlsx"
CLEAN_PATH = DATA_DIR / "compressor_clean.parquet"

KEEP_COLS = ["T_e", "T_c", "Capacity", "Power"]

RANGE_RULES = {
    "T_e": (-35, 15),
    "T_c": (20, 70),
    "Capacity": (0, None),
    "Power": (0, None),
}


def download_raw():
    if RAW_PATH.exists():
        print(f"  Raw file already present: {RAW_PATH}")
        return
    import requests
    print(f"  Downloading {RAW_URL} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(RAW_URL, timeout=30)
    r.raise_for_status()
    RAW_PATH.write_bytes(r.content)
    print(f"  Saved to {RAW_PATH}")


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dropped = {}
    for col, (lo, hi) in RANGE_RULES.items():
        if lo is not None and hi is not None:
            mask = ~df[col].between(lo, hi, inclusive="both")
            reason = f"{col} outside [{lo}, {hi}]"
        elif lo is not None:
            mask = df[col] <= lo
            reason = f"{col} <= {lo}"
        else:
            mask = df[col] <= (hi or 0)
            reason = f"{col} <= 0"

        bad = df[mask]
        if not bad.empty:
            dropped[reason] = len(bad)
            df = df[~mask]
    return df.reset_index(drop=True), dropped


def main():
    print("=== Stage 1: Ingest ===")

    download_raw()

    print("  Loading Excel ...")
    raw = pd.read_excel(RAW_PATH)
    n_in = len(raw)
    print(f"  Rows in: {n_in}")

    df = raw[KEEP_COLS].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    n_nan = df.isna().any(axis=1).sum()
    if n_nan:
        print(f"  Dropped {n_nan} rows with non-numeric values")
        df = df.dropna()

    df, dropped = validate(df)

    total_dropped = n_nan + sum(dropped.values())
    for reason, count in dropped.items():
        print(f"  Dropped {count} row(s): {reason}")
    print(f"  Total rows dropped: {total_dropped}")
    print(f"  Rows out: {len(df)}")

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PATH, index=False)
    print(f"  Written: {CLEAN_PATH}")
    print("=== Ingest complete ===\n")


if __name__ == "__main__":
    main()

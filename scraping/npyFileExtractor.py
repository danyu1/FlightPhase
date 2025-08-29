import argparse
import os
import numpy as np
import pandas as pd

DEFAULT_SHOW_COLS = [
    "SeasonYear", "SeasonLabel", "Gender", "Division", "School",
    "Team", "Event", "Athlete", "ClassYear", "MarkOrTime",
    "Wind", "Meet", "MeetDate", "SourceURL"
]

def load_structured_npy(path: str) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.size == 0:
        return pd.DataFrame(columns=DEFAULT_SHOW_COLS)
    df = pd.DataFrame.from_records(arr)

    # Normalize dtypes
    for c in df.columns:
        if df[c].dtype.kind in ("U", "S", "O"):
            df[c] = df[c].astype("string").fillna("")
    for c in ("SeasonYear", "ListHnd", "SeasonHnd"):
        if c in df.columns:
            # -1 means missing (from scraper); keep as Int32 for nullability
            df[c] = pd.Series(df[c], dtype="Int32")
    return df

def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df
    if args.year is not None and "SeasonYear" in out.columns:
        out = out[out["SeasonYear"] == args.year]
    if args.season and "SeasonLabel" in out.columns:
        out = out[out["SeasonLabel"].str.casefold() == args.season.casefold()]
    if args.gender and "Gender" in out.columns:
        out = out[out["Gender"].str.casefold() == args.gender.casefold()]
    if args.school and "School" in out.columns:
        out = out[out["School"].str.contains(args.school, case=False, na=False)]
    if args.event and "Event" in out.columns:
        out = out[out["Event"].str.contains(args.event, case=False, na=False)]
    if args.athlete and "Athlete" in out.columns:
        out = out[out["Athlete"].str.contains(args.athlete, case=False, na=False)]
    return out

def print_summary(df: pd.DataFrame, title: str, max_rows: int):
    print(f"\n=== {title} ===")
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    if df.empty:
        return

    cols_to_show = [c for c in DEFAULT_SHOW_COLS if c in df.columns]
    print("\nSample rows:")
    print(df[cols_to_show].head(max_rows).to_string(index=False))

    # Simple breakdowns (guarded if columns exist)
    if {"SeasonYear","SeasonLabel","Gender"}.issubset(df.columns):
        print("\nCounts by Year × Season × Gender (top 15):")
        counts = (df.groupby(["SeasonYear","SeasonLabel","Gender"])
                    .size().rename("count").reset_index()
                    .sort_values("count", ascending=False)
                    .head(15))
        print(counts.to_string(index=False))

    if "Event" in df.columns:
        print("\nTop Events (by rows):")
        print(df["Event"].value_counts().head(50).to_string())

    if "School" in df.columns:
        print("\nTop Schools (by rows):")
        print(df["School"].value_counts().head(10).to_string())

def main():
    ap = argparse.ArgumentParser(
        description="Inspect and filter a TFRRS .npy structured array."
    )
    ap.add_argument("-i", "--input", default="tfrrs_performances_fast.ckpt_22000.npy",
                    help="Path to .npy file (structured array)")
    ap.add_argument("-o", "--output-csv", default=None,
                    help="Optional path to write filtered rows as CSV")
    ap.add_argument("--year", type=int, help="Filter: SeasonYear (e.g., 2025)")
    ap.add_argument("--season", choices=["Indoors","Outdoors"],
                    help="Filter: SeasonLabel")
    ap.add_argument("--gender", choices=["Men","Women"],
                    help="Filter: Gender")
    ap.add_argument("--school", help="Filter: substring match on School")
    ap.add_argument("--event", help="Filter: substring match on Event")
    ap.add_argument("--athlete", help="Filter: substring match on Athlete")
    ap.add_argument("--head", type=int, default=10,
                    help="How many sample rows to print (default: 10)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"File not found: {args.input}")

    df = load_structured_npy(args.input)

    # Global summary (unfiltered)
    print_summary(df, "Dataset summary (all rows)", args.head)

    # Filtered view (if any filters supplied)
    filtered = apply_filters(df, args)
    if any(getattr(args, k) is not None for k in ["year","season","gender","school","event","athlete"]):
        print_summary(filtered, "Filtered summary", args.head)

    # Optional export
    if args.output_csv:
        filtered.to_csv(args.output_csv, index=False)
        print(f"\n✅ Wrote {len(filtered):,} filtered rows to {args.output_csv}")

if __name__ == "__main__":
    main()

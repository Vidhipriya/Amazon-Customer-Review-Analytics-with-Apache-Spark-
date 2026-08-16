from pathlib import Path
import time

import pandas as pd

# Define project folders relative to the current script location.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "amazon_reviews_multilingual_US_v1_00.tsv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Set the maximum number of rows for the single-machine pandas comparison.
SAMPLE_SIZE = 100_000


def main():
    # Start runtime measurement for the pandas sample workflow.
    start_time = time.time()

    # Retain only fields required for rating and verified-purchase comparison.
    use_columns = [
        "review_id",
        "product_id",
        "star_rating",
        "helpful_votes",
        "total_votes",
        "verified_purchase",
        "review_date",
    ]

    # Read only the first 100,000 TSV records into one local pandas DataFrame.
    # This is a sequential sample, not a random sample of the full dataset.
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        usecols=use_columns,
        nrows=SAMPLE_SIZE,
        low_memory=False
    )

    # Convert rating and vote fields from text to numeric values.
    # Invalid values are converted to missing values for later validation.
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(
        df["helpful_votes"],
        errors="coerce"
    ).fillna(0)

    df["total_votes"] = pd.to_numeric(
        df["total_votes"],
        errors="coerce"
    ).fillna(0)

    # Convert the review-date field from text to datetime format.
    df["review_date"] = pd.to_datetime(
        df["review_date"],
        errors="coerce"
    )

    # Retain valid records with identifiers, a valid 1–5 rating, and a date.
    df = df[
        df["review_id"].notna()
        & df["product_id"].notna()
        & df["star_rating"].between(1, 5)
        & df["review_date"].notna()
    ].copy()

    # Calculate helpfulness only for reviews that received at least one vote.
    df["helpfulness_ratio"] = (
        df["helpful_votes"] / df["total_votes"]
    ).where(df["total_votes"] > 0)

    # Count the number of sample reviews for each star-rating value.
    rating_distribution = (
        df.groupby("star_rating")
        .size()
        .reset_index(name="review_count")
    )

    # Compare review volume, mean rating, and mean helpfulness by verification status.
    verified_analysis = (
        df.groupby("verified_purchase")
        .agg(
            review_count=("review_id", "count"),
            average_rating=("star_rating", "mean"),
            average_helpfulness=("helpfulness_ratio", "mean")
        )
        .reset_index()
    )

    # Export small pandas aggregate tables for report comparison.
    rating_distribution.to_csv(
        OUTPUT_DIR / "pandas_100k_rating_distribution.csv",
        index=False
    )

    verified_analysis.to_csv(
        OUTPUT_DIR / "pandas_100k_verified_analysis.csv",
        index=False
    )

    # Calculate total time used to read, clean, aggregate, and export the sample.
    elapsed_seconds = time.time() - start_time

    # Save runtime information for the Spark-versus-pandas comparison table.
    with open(OUTPUT_DIR / "pandas_100k_runtime.txt", "w") as file:
        file.write(f"Sample size: {SAMPLE_SIZE:,} rows\n")
        file.write(f"Valid records: {len(df):,}\n")
        file.write(f"Runtime seconds: {elapsed_seconds:.2f}\n")

    # Display key results in the terminal for verification and screenshots.
    print(f"Pandas rows read: {SAMPLE_SIZE:,}")
    print(f"Valid pandas records: {len(df):,}")
    print(f"Runtime: {elapsed_seconds:.2f} seconds")
    print("\nRating distribution:")
    print(rating_distribution.to_string(index=False))
    print("\nVerified-purchase analysis:")
    print(verified_analysis.to_string(index=False))

# Run the pandas comparison only when this file is executed directly.
if __name__ == "__main__":
    main()
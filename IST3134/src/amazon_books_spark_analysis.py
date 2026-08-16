from pathlib import Path
import time

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F

# Define project folders relative to this script's location.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "amazon_reviews_multilingual_US_v1_00.tsv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Create the output folder if it does not exist.
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # Create a local Spark session with memory-safe settings for the large TSV file.
    spark = (
        SparkSession.builder
        .appName("AmazonBooksReviewsAnalysis")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    # Reduce non-essential Spark messages and start performance timing.
    spark.sparkContext.setLogLevel("WARN")
    start_time = time.time()

    print("\nLoading Amazon review data...")

    # Read the TSV file and retain only fields required for structured analysis.
    raw_reviews = (
        spark.read
        .option("header", True)
        .option("sep", "\t")
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(DATA_PATH))
        .select(
            "customer_id",
            "review_id",
            "product_id",
            "product_title",
            "star_rating",
            "helpful_votes",
            "total_votes",
            "verified_purchase",
            "review_date"
        )
    )

    # Count all records before validation and filtering.
    raw_count = raw_reviews.count()

    # Clean data, convert types, filter invalid records, and derive analysis variables.
    reviews = (
        raw_reviews
        .withColumn("star_rating", F.col("star_rating").cast("int"))
        .withColumn("helpful_votes", F.col("helpful_votes").cast("int"))
        .withColumn("total_votes", F.col("total_votes").cast("int"))
        .withColumn("review_date", F.to_date("review_date"))
        .filter(F.col("review_id").isNotNull())
        .filter(F.col("product_id").isNotNull())
        .filter(F.col("star_rating").between(1, 5))
        .filter(F.col("review_date").isNotNull())
        .fillna({"helpful_votes": 0, "total_votes": 0})
        .withColumn(
            # Calculate the proportion of votes that marked a review as helpful.
            "helpfulness_ratio",
            F.when(
                F.col("total_votes") > 0,
                F.round(F.col("helpful_votes") / F.col("total_votes"), 3)
            ).otherwise(None)
        )
        .withColumn(
            # Create transparent rating-derived sentiment labels.
            "sentiment_label",
            F.when(F.col("star_rating") >= 4, "Positive")
            .when(F.col("star_rating") == 3, "Neutral")
            .otherwise("Negative")
        )
    )

    # Trigger validation and obtain the number of valid review records.
    clean_count = reviews.count()

    print("\n--- Dataset quality summary ---")
    print(f"Raw review count: {raw_count:,}")
    print(f"Valid review count: {clean_count:,}")
    print(f"Invalid or incomplete records removed: {raw_count - clean_count:,}")
    print("Note: Full duplicate removal was omitted to keep the local Spark run memory-safe.")

    # Produce statistics for the cleaned dataset.
    overview = reviews.agg(
        F.count("*").alias("total_reviews"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.countDistinct("product_id").alias("unique_products"),
        F.min("review_date").alias("earliest_review"),
        F.max("review_date").alias("latest_review"),
        F.round(F.avg("star_rating"), 2).alias("overall_average_rating")
    )

    print("\n--- Dataset overview ---")
    overview.show(truncate=False)

    # Count reviews for each rating and calculate its proportion of all valid reviews.
    rating_distribution = (
        reviews
        .groupBy("star_rating")
        .count()
        .withColumn(
            "percentage",
            F.round(F.col("count") / F.lit(clean_count) * 100, 2)
        )
        .orderBy("star_rating")
    )

    print("\n--- Rating distribution ---")
    rating_distribution.show(truncate=False)

    # Summarise the Positive, Neutral, and Negative rating-derived sentiment groups.
    sentiment_summary = (
        reviews
        .groupBy("sentiment_label")
        .count()
        .withColumn(
            "percentage",
            F.round(F.col("count") / F.lit(clean_count) * 100, 2)
        )
        .orderBy(
            F.when(F.col("sentiment_label") == "Positive", 1)
            .when(F.col("sentiment_label") == "Neutral", 2)
            .otherwise(3)
        )
    )

    print("\n--- Rating-derived sentiment summary ---")
    sentiment_summary.show(truncate=False)

    # Compare review volume, ratings, helpfulness, and positivity by purchase verification.
    verified_analysis = (
        reviews
        .groupBy("verified_purchase")
        .agg(
            F.count("*").alias("review_count"),
            F.round(F.avg("star_rating"), 2).alias("average_rating"),
            F.round(F.avg("helpfulness_ratio"), 3).alias("average_helpfulness"),
            F.round(
                F.avg(
                    F.when(F.col("star_rating") >= 4, 1).otherwise(0)
                ) * 100,
                2
            ).alias("positive_review_percent")
        )
        .orderBy("verified_purchase")
    )

    print("\n--- Verified-purchase comparison ---")
    verified_analysis.show(truncate=False)

    # Analyse helpfulness by rating only for reviews with at least 10 total votes.
    helpful_reviews = (
        reviews
        .filter(F.col("total_votes") >= 10)
        .groupBy("star_rating")
        .agg(
            F.count("*").alias("review_count"),
            F.round(F.avg("helpfulness_ratio"), 3).alias(
                "average_helpfulness"
            )
        )
        .orderBy("star_rating")
    )

    print("\n--- Helpfulness by rating: reviews with 10+ votes ---")
    helpful_reviews.show(truncate=False)

    # Aggregate total review volume and average rating for each year-month.
    monthly_trend = (
        reviews
        .withColumn("year_month", F.date_format("review_date", "yyyy-MM"))
        .groupBy("year_month")
        .agg(
            F.count("*").alias("review_count"),
            F.round(F.avg("star_rating"), 2).alias("average_rating")
        )
        .orderBy("year_month")
    )

    print("\n--- Monthly review trend ---")
    monthly_trend.show(50, truncate=False)

    # Select a reproducible one-seventh hash-based sample for product-level aggregation.
    # Sampling limits memory use because product title grouping has very high cardinality.
    product_sample = (
        reviews
        .filter(F.pmod(F.hash("review_id"), F.lit(7)) == 0)
        .select(
            "product_id",
            "product_title",
            "star_rating",
            "helpful_votes",
            "helpfulness_ratio"
        )
    )

    # Identify the 20 highest-volume products in the reproducible sample.
    top_products = (
        product_sample
        .groupBy("product_id", "product_title")
        .agg(
            F.count("*").alias("sample_review_count"),
            F.round(F.avg("star_rating"), 2).alias("average_rating"),
            F.sum("helpful_votes").alias("total_helpful_votes"),
            F.round(F.avg("helpfulness_ratio"), 3).alias(
                "average_helpfulness"
            )
        )
        .filter(F.col("sample_review_count") >= 20)
        .orderBy(F.desc("sample_review_count"))
        .limit(20)
    )

    print("\n--- Top products by review volume: reproducible 1/7 sample ---")
    top_products.show(truncate=False)

    reviews.createOrReplaceTempView("amazon_reviews")

    print("\n--- Spark SQL: verified-purchase rating summary ---")

    # Run the verified-purchase aggregation using Spark SQL.
    spark.sql("""
        SELECT
            verified_purchase,
            COUNT(*) AS review_count,
            ROUND(AVG(star_rating), 2) AS average_rating,
            ROUND(AVG(helpfulness_ratio), 3) AS average_helpfulness
        FROM amazon_reviews
        GROUP BY verified_purchase
        ORDER BY verified_purchase
    """).show(truncate=False)

    # Convert only small aggregated result tables to pandas for export and visualisation.
    rating_pd = rating_distribution.toPandas()
    sentiment_pd = sentiment_summary.toPandas()
    verified_pd = verified_analysis.toPandas()
    helpful_pd = helpful_reviews.toPandas()
    monthly_pd = monthly_trend.toPandas()
    top_products_pd = top_products.toPandas()

    # Export the final aggregate result tables as CSV files.
    rating_pd.to_csv(
        OUTPUT_DIR / "rating_distribution.csv",
        index=False
    )

    sentiment_pd.to_csv(
        OUTPUT_DIR / "sentiment_summary.csv",
        index=False
    )

    verified_pd.to_csv(
        OUTPUT_DIR / "verified_purchase_analysis.csv",
        index=False
    )

    helpful_pd.to_csv(
        OUTPUT_DIR / "helpfulness_by_rating.csv",
        index=False
    )

    monthly_pd.to_csv(
        OUTPUT_DIR / "monthly_review_trend.csv",
        index=False
    )

    top_products_pd.to_csv(
        OUTPUT_DIR / "top_products_sample.csv",
        index=False
    )

    # Create and save the chart showing the number of reviews for each star rating.
    plt.figure(figsize=(8, 5))
    plt.bar(
        rating_pd["star_rating"].astype(str),
        rating_pd["count"],
        color="#4C78A8"
    )
    plt.xlabel("Star rating")
    plt.ylabel("Number of reviews")
    plt.title("Distribution of Amazon Review Ratings")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "rating_distribution.jpg",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    bar_colours = ["#F58518", "#54A24B"]

    plt.bar(
        verified_pd["verified_purchase"].astype(str),
        verified_pd["average_rating"],
        color=bar_colours
    )
    plt.xlabel("Verified purchase")
    plt.ylabel("Average star rating")
    plt.title("Average Rating by Verified-Purchase Status")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "verified_purchase_average_rating.jpg",
        dpi=300
    )
    plt.close()

    # Convert monthly values to strings for use as x-axis labels.
    monthly_pd["year_month"] = monthly_pd["year_month"].astype(str)

    # Create a monthly rating trend chart with one readable label per year.
    plt.figure(figsize=(12, 5))
    plt.plot(
        monthly_pd["year_month"],
        monthly_pd["average_rating"],
        linewidth=1.5,
        color="#E45756"
    )

    # Display every twelfth month to avoid overcrowding the x-axis.
    year_tick_positions = list(range(0, len(monthly_pd), 12))
    year_tick_labels = monthly_pd["year_month"].iloc[::12].tolist()

    plt.xticks(
        ticks=year_tick_positions,
        labels=year_tick_labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel("Year")
    plt.ylabel("Average star rating")
    plt.title("Monthly Average Rating of Amazon Reviews")
    plt.ylim(3.9, 5.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "monthly_average_rating_improved.jpg",
        dpi=300
    )
    plt.close()

    # Print final runtime and stop the Spark session to release resources.
    print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds.")
    print(f"Outputs saved in: {OUTPUT_DIR}")

    spark.stop()

# Run the workflow only when this file is executed directly.
if __name__ == "__main__":
    main()
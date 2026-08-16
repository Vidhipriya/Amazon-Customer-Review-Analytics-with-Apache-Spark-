# Scaling Amazon Customer Review Analytics with Apache Spark

## Project Overview

This project implements a Big Data Analytics workflow using **Apache Spark**, **PySpark DataFrames**, and **Spark SQL** to analyse Amazon customer reviews. The analysis focuses on rating distribution, rating-derived sentiment, verified-purchase behaviour, review helpfulness, monthly rating trends, and highly reviewed products.

The project also includes a conventional **pandas** implementation on the first 100,000 records of the TSV file. This provides a single-machine comparison with the full-dataset Spark workflow.

## Research Questions

1. What is the overall distribution of the star ratings (1-5) from the data set and how can this distribution be best summarised as Positive / Neutral / Negative sentiment?
2. Does this mean that verified purchases have different average ratings, helpfulness scores, and positive-review rates than unverified purchases?
3. Do higher-rated reviews receive higher helpfulness scores than lower-rated reviews, and if so, how does this helpfulness votes are cast?
4. Has the average star rating and the number of reviews per month for each month changed from 1995 to 2015?
5. Which products receive the most reviews and what is the average rating and helpfulness of the reviews?

## Dataset

- **Source:** [Amazon US Customer Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset)
- **Input file:** `amazon_reviews_multilingual_US_v1_00.tsv`
- **Format:** Tab-separated values (TSV)
- **Raw records loaded:** 6,931,166
- **Valid records analysed by Spark:** 6,930,546
- **Columns in source file:** 15

The raw TSV file is not included in this repository because of its size. Download the dataset from Kaggle and place the selected file in the following location:

```text
data/amazon_reviews_multilingual_US_v1_00.tsv
```

## Technologies

- Python
- Apache Spark / PySpark 4.2.0
- Spark SQL
- pandas
- Matplotlib
- Java Development Kit 17 or later
- Visual Studio Code

## Project Structure

```text
BigDataAnalyticProject/
├── outputs/
│   ├── rating_distribution.csv
│   ├── sentiment_summary.csv
│   ├── verified_purchase_analysis.csv
│   ├── helpfulness_by_rating.csv
│   ├── monthly_review_trend.csv
│   ├── top_products_sample.csv
│   ├── pandas_100k_rating_distribution.csv
│   ├── pandas_100k_verified_analysis.csv
│   ├── pandas_100k_runtime.txt
│   ├── rating_distribution.jpg
│   ├── verified_purchase_average_rating.jpg
│   └── monthly_average_rating_improved.jpg
├── src/
│   ├── amazon_books_spark_analysis.py
│   └── pandas_sample_comparison.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Java and Spark

Spark requires Java 17 or later. Confirm Java is available:

```powershell
java -version
```

Test the PySpark installation:

```powershell
python -c "from pyspark.sql import SparkSession; spark=SparkSession.builder.appName('SparkTest').getOrCreate(); print('Spark version:', spark.version); spark.stop()"
```

## Run the Spark Analysis

Run the full structured-data Spark analysis:

```powershell
python src\amazon_books_spark_analysis.py
```

The Spark pipeline performs the following tasks:

1. Reads the TSV file using the tab delimiter.
2. Selects required structured variables.
3. Converts rating, vote, and date fields to appropriate types.
4. Removes invalid or incomplete review records.
5. Creates `helpfulness_ratio` and rating-derived `sentiment_label` variables.
6. Produces rating-distribution, sentiment, verified-purchase, helpfulness, and monthly-trend aggregates on the full valid dataset.
7. Creates a reproducible one-seventh hash-based sample for high-cardinality product-level analysis.
8. Executes an equivalent verified-purchase summary using Spark SQL.
9. Saves aggregate CSV outputs and JPG figures under `outputs/`.

### Rating-derived sentiment definition

| Star rating | Sentiment label |
|---:|---|
| 4–5 | Positive |
| 3 | Neutral |
| 1–2 | Negative |

This is a transparent rule-based label derived from star ratings. It is **not** text-based NLP sentiment classification.

### Helpfulness definition

```text
helpfulness_ratio = helpful_votes / total_votes
```

The helpfulness-by-rating analysis includes only reviews with at least 10 total votes.

## Run the pandas Comparison

Run the conventional single-machine comparison:

```powershell
python src\pandas_sample_comparison.py
```

This script reads the **first 100,000 records** from the TSV file. It is a sequential file sample, not a random sample. It exports rating and verified-purchase summaries, as well as runtime information, to the `outputs/` folder.

## Spark Processing Logic

The PySpark workflow follows a MapReduce-like logical pattern:

```text
TSV input partitions
        ↓
Map-like transformations
- Select columns
- Convert types
- Filter invalid records
- Derive helpfulness and sentiment fields
        ↓
Shuffle/group by key
- star_rating
- sentiment_label
- verified_purchase
- year_month
        ↓
Reduce-like aggregation
- COUNT
- AVG
- SUM
        ↓
CSV output tables and visualisations
```

For example, rating distribution conceptually maps each review to `(star_rating, 1)`, groups records by `star_rating`, and sums the values to calculate the number of reviews at each rating.

## Key Results

Results from the full Spark analysis include:

| Finding | Result |
|---|---:|
| Valid reviews analysed | 6,930,546 |
| Five-star reviews | 4,441,511 (64.09%) |
| Positive reviews, 4–5 stars | 5,707,732 (82.36%) |
| Average rating: verified purchases | 4.37 |
| Average rating: non-verified purchases | 4.14 |
| Average helpfulness: verified purchases | 0.478 |
| Average helpfulness: non-verified purchases | 0.565 |
| pandas runtime for first 100,000 records | 0.87 seconds |

## Limitations

- Spark was executed in local mode on a laptop rather than on a multi-node cloud cluster.
- The full-data duplicate-removal operation was not included in the memory-safe local pipeline.
- Product-level aggregation uses a reproducible one-seventh hash-based sample because grouping by product title has high cardinality.
- The pandas comparison uses the first 100,000 records, so it can be temporally biased and is not a random sample.
- Sentiment labels are based on star ratings rather than review-text classification.
- Helpfulness scores are observational and can be influenced by exposure, product popularity, review length, and review age.


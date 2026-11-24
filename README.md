# Netflix Content Analytics

End-to-end data analytics and machine learning project exploring the Netflix catalog (`8,783` unique titles), covering data cleaning, exploratory data analysis, content-based recommendation modeling, and time-series content trend forecasting.

---

## Repository Structure
```
Netflix-Content-Analytics/
├── data/
│   ├── Dataset.csv                        # Raw Netflix catalog dataset
│   └── df_clean.csv                       # Cleaned dataset (8,783 rows × 16 columns)
├── task1_data_cleaning/
│   └── task1_data_cleaning.ipynb          # Schema profiling, cleaning, and feature engineering
├── task2_eda/
│   └── task2_eda.ipynb                    # Catalog distribution, temporal trends, and key metrics
├── task3_recommendation/
│   └── task3_recommendation.ipynb         # TF-IDF & cosine similarity content-based recommender
├── task4_trend_prediction/
│   └── task4_trend_prediction.ipynb       # Time-series trend modeling, backtesting, and forecasting
└── README.md
```
---

## Project Workflow & Key Quantitative Results

### 1. Data Cleaning & Feature Engineering (`task1_data_cleaning`)
- **Dataset Standardization:** Cleaned and reshaped the raw catalog into `data/df_clean.csv` (**8,783** unique titles across **16** columns), preserving genuine missingness (**2,586** missing `director` entries at **29.44%** and **287** missing `country` entries at **3.27%**) as `NaN` rather than imputing synthetic labels.
- **Feature Extraction:** Converted `date_added` to datetime (`2008-01-01` to `2021-09-25`) to derive `year_added` and `month_added`; split `duration` into numeric `duration_min` (movies) and `seasons` (TV shows); and extracted `primary_category` and `category_count` from `listed_in`.

### 2. Exploratory Data Analysis & Key Metrics (`task2_eda`)
- **Catalog Composition:** Movies account for **6,121** titles (**69.69%**) versus **2,662** TV Shows (**30.31%**).
- **Temporal Growth:** Catalog additions accelerated after **2016** and peaked in **2019** (**2,012** titles), followed by **1,879** in **2020** and **1,498** in **2021** (partial year through September, where TV Show share reached **33.71%**).
- **Geographic & Genre Distribution:** Production is led by the **United States** (**3,239** titles, **38.12%** of known countries), **India** (**1,056**, **12.43%**), and the **United Kingdom** (**638**, **7.51%**). Across all listed genre tags, **International Movies** leads (**2,749** titles, **31.30%**), followed by **Dramas** (**2,422**, **27.58%**) and **Comedies** (**1,671**, **19.03%**).
- **Ratings & Runtime Metrics:** Mature ratings dominate the catalog (**TV-MA**: **3,202** titles; **TV-14**: **2,155**). Movie runtimes average **99.6 min** (median **98.0 min**), while **1,790** TV Shows (**67.24%**) consist of a single season.

### 3. Content-Based Recommendation System (`task3_recommendation`)
- **Vectorization & Similarity Modeling:** Encoded **5** metadata attributes (`type`, `rating`, `listed_in`, `country`, `director`) into whitespace-delimited tokens (`content_profile`), generating an **(8,783 × 5,130)** TF-IDF feature matrix and an **(8,783 × 8,783)** pairwise cosine similarity matrix (**77,141,089** comparisons) while leaving missing fields empty to prevent false similarity.
- **Verified Performance & Edge-Case Handling:** Across all **8,783** titles, top-1 same-type recommendations achieve a mean cosine similarity of **0.7219** (median **0.8142**), with **5,879** titles (**66.94%**) matching at **≥ 0.50**. Validated across thematic, regional, and edge-case queries (*Midnight Mass* top matches at **0.7419**; *Sacred Games* Indian crime dramas up to **0.5389**; exact vs. case-insensitive separation of *DEATH NOTE* TV anime at **1.0000** vs. *Death Note* feature film at **0.4007**; and partial-match fallback for unknown queries).

### 4. Content Trend Prediction (`task4_trend_prediction`)
- **Time-Series Modeling & Backtesting:** Aggregated catalog additions into a continuous **165-month** series (`2008-01` to `2021-09`), isolating the **69-month** modern streaming era (`2016-01` to `2021-09`, **8,645** titles / **98.43%** of the catalog) and accounting for `2021` containing **9 recorded months** (`1,498` titles). Across a chronological **57-month train** (`2016-01` to `2020-09`) / **12-month test** (`2020-10` to `2021-09`) split, `Log-Growth Trend` achieved the lowest forecasting error on total additions (**MAE = 29.56**, **RMSE = 39.62**), beating the `Trailing 12M Mean` baseline (**MAE = 31.65**) and cutting MAE by **45.1%** compared to `Linear Trend` (**MAE = 53.82**, **RMSE = 60.78**).
- **2021–2023 Content Projections:** Refitting `Log-Growth Trend` on all **69** modern months projects **527** titles in `2021 Q4` (`368` Movies, `159` TV Shows), bringing full-year **2021** to **2,025** titles (`1,361` Movies, `664` TV Shows; **32.78%** TV share). Annual additions are projected at **2,171** titles in **2022** (`1,517` Movies, `654` TV Shows; **30.12%** TV share) and **2,258** titles in **2023** (`1,578` Movies, `680` TV Shows; **30.10%** TV share), with monthly residual standard deviations of **26.21** (`Movies`) and **11.69** (`TV Shows`).

---

## Tech Stack

- **Language & Core Libraries:** Python 3.13, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning & NLP:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`, `LinearRegression`, `Ridge`)
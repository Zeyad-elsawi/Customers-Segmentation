# Customer Segmentation Notebook (Segmentation.ipynb)

## Purpose
Discover distinct customer/product behavior groups from transactional data to power personalization, targeting, and analytics.

## Data
- Input: data.csv (Online Retail-style schema)
  - InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- Size after cleaning: ~406,829 rows

## Environment
- Python 3.10+
- Packages: pandas, numpy, seaborn, matplotlib, scikit-learn

Install:
```bash
pip install -r requirements.txt
```

## How to Run
1) Launch Jupyter:
```bash
jupyter notebook Segmentation.ipynb
```
2) Run cells top-to-bottom.

## Pipeline (Notebook Sections)
1) Load data
   - Reads data.csv, basic info/head

2) Clean data
   - Drop rows with missing CustomerID/Description
   - Remove negative quantities
   - Reset index

3) Encode features
   - LabelEncode: InvoiceNo, StockCode, Description, InvoiceDate, CustomerID, Country
   - Keep numeric: Quantity, UnitPrice

4) Scale features
   - MinMaxScaler(feature_range=(1,5))
   - normalized_df used for clustering and PCA

5) Explore distributions
   - Histogram of all features (sanity check, skew/outliers)

6) Dimensionality Reduction (PCA)
   - Quick 2D scatter for structure
   - Variance explained for 2–5 components

7) Choose K (Elbow)
   - Iterate K=1..9, plot inertia to pick K
   - In this notebook: KMeans(n_clusters=5)

8) Cluster
   - Fit KMeans on normalized_df
   - Predict labels and attach as df1['Cluster']

9) Visualize & Profile
   - PCA 3D/2D coloring by cluster
   - Cluster counts/fractions
   - Use describe()/groupby to inspect per-cluster stats (recommended extension)

## Key Parameters You Can Tweak
- Scaling: MinMaxScaler(range=(1,5)) → StandardScaler() if needed
- Features: remove/add columns (e.g., drop text-heavy encodings if noisy)
- KMeans(n_clusters=5): change K via elbow plot
- Alternative clustering: DBSCAN, GMM, MeanShift (imports included)

## Outputs
- In-notebook artifacts:
  - df1 with new column Cluster (0..K-1)
  - PCA plots
  - Cluster size distribution
- Save clusters (optional):
```python
df1.to_csv('segmented_customers.csv', index=False)
```

## Interpreting Clusters
- Use groupby to build a concise profile per cluster:
```python
df1.groupby('Cluster')[['Quantity','UnitPrice','CustomerID','Country']].agg(['mean','median'])
```
- Suggested profiling:
  - Recency/Frequency/Monetary (RFM) derived from InvoiceDate, InvoiceNo, UnitPrice, Quantity
  - Country mix, average basket value, product-code concentration

## Using Segments Downstream
- Attach cluster id to users/sessions for:
  - Targeted discounts and messaging
  - Personalized recommendations
  - Lookalike modeling and cohort analysis
- Export a mapping:
```python
df1[['CustomerID','Cluster']].drop_duplicates().to_csv('customer_segments.csv', index=False)
```

## Tips & Troubleshooting
- Memory: If you hit memory issues, sample:
```python
normalized_df_sample = normalized_df.sample(100000, random_state=0)
```
- K selection: Validate with silhouette score:
```python
from sklearn.metrics import silhouette_score
silhouette_score(normalized_df, labels)
```
- Noisy encodings: LabelEncoding of text ids can inject arbitrary ordinality. Consider:
  - Dropping overly high-cardinality columns
  - Aggregating to customer-level features first (e.g., per-customer RFM), then cluster

## Extensions (Recommended)
- Customer-level aggregation (RFM) before clustering
- Feature selection (variance threshold or correlation pruning)
- PCA/UMAP for better separability
- Persist scaler and cluster model for reuse

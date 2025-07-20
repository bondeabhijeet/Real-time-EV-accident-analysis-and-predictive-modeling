## Directory Structure

### `final_data/`
- Contains the final cleaned and processed CSV datasets ready for modeling and analysis.

### `Intermediate/`, `Intermediate2/`, `Intermediate3/`
- Store intermediate CSV files generated during different stages of the data cleaning process.

### `SparkApplicationUI/`
- Includes PDF exports from the Spark Application UI (`localhost:4040`) for job execution tracking and DAG visualization.

## Notebooks

### `DataCleaning1_PySpark.ipynb` to `DataCleaning4_PySpark.ipynb`
- Jupyter notebooks for cleaning and transforming the raw dataset across multiple stages using PySpark.

### `ev_accident_analysis_classification_pyspark.ipynb`
- Applies classification machine learning models on the processed dataset (e.g., Random Forest, Logistic Regression) to predict accident severity.

### `ev_accident_regression_pyspark.ipynb`
- Implements regression models to estimate numeric outcomes like injury counts or property damage using PySpark MLlib.

## Dataset

### `Motor_Vehicle_Collisions_-_Crashes.csv`
- The original raw dataset containing vehicle collision records, which serves as the starting point for all data preprocessing and analysis.

---
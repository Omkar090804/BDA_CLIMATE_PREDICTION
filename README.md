# 🌍 BDA_CLIMATE_PREDICTION

## 📜 Project Description

This project demonstrates the use of **Big Data Analytics** to process and analyze large-scale climate data in order to predict the **effects of climate change**, such as **temperature and sea level changes**.

By leveraging a suite of big data tools, the project transforms raw climate records into a structured format suitable for **machine learning**, ultimately building predictive models to forecast **future climate trends**.

The **primary goal** is to predict temperature changes using **predictive modeling** on climate data with **Apache Spark**.  
The framework developed is **scalable**, **cost-effective**, and **efficient** compared to existing solutions.

---

## 📊 Dataset

- **Dataset:** Global Historical Climatology Network (GHCN-Daily)  
- **Source:** [NOAA (National Oceanic and Atmospheric Administration)](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)  
- **File Used:** `AE000041196.csv`  
- **Observation Station:** AE000041196 (United Arab Emirates)

### Dataset Features:
- `TMAX` – Maximum Temperature  
- `TMIN` – Minimum Temperature  
- `TAVG` – Average Temperature  
- `PRCP` – Precipitation  

The dataset contains **daily weather records** over a long time period.

---

## 🛠️ Technology Stack

| Category | Tool |
|-----------|------|
| **Storage** | Hadoop Distributed File System (HDFS) |
| **Processing Engine** | Apache Spark |
| **Querying Tool** | Apache Hive |
| **Machine Learning** | Spark MLlib, Apache Mahout |
| **Programming Language** | Python 3.10.12 (PySpark) |
| **Core Libraries** | Pandas, Matplotlib, Seaborn, NumPy |

---

## ⚙️ Methodology

The project follows a complete pipeline from **data ingestion** to **predictive modeling** and **report generation**.

### 1️⃣ Data Preprocessing

- **Data Cleaning:** Removed records with null or missing values in critical columns (`DATE`, `TAVG`, `LATITUDE`, `LONGITUDE`).  
- **Unit Conversion:**  
  - Temperature values (`TMAX`, `TMIN`, `TAVG`) converted from tenths of °C → °C.  
  - Precipitation (`PRCP`) converted from tenths of mm → mm.

### 2️⃣ Feature Engineering

To enhance the dataset for predictive modeling:
- **Temporal Features:** Extracted `YEAR`, `MONTH`, and `DAY_OF_YEAR` from the `DATE` column.  
- **Categorical Features:**  
  - Created `SEASON` based on the month.  
  - Added `DECADE` for long-term trend analysis.  
- **Calculated Features:**  
  - `TEMP_RANGE = TMAX - TMIN`  
  - `TAVG_30DAY_AVG` (30-day moving average of temperature).

### 3️⃣ Handling Missing Data

- Missing `PRCP` values filled with `0`.  
- Missing `TMAX` and `TMIN` values imputed using:  
  - `TMAX = TAVG + 5°C`  
  - `TMIN = TAVG - 5°C`

### 4️⃣ Machine Learning Modeling

Models Implemented:
1. Linear Regression  
2. Random Forest Regressor  
3. Gradient Boosted Trees (GBT) Regressor  

- **Data Split:** 70% Training | 30% Testing  
- **Evaluation Metrics:** RMSE, MAE, R²  

---

## 📈 Results

### 🔹 Key Findings

- Average temperature increase of **+2.4°C** observed over **40 years**.  
- **Random Forest** model achieved the highest predictive accuracy.

| Model | RMSE | MAE | R² | Adjusted R² | MAPE | Accuracy |
|--------|------|------|------|--------------|-------|-----------|
| Linear Regression | 0.9659 | 0.7386 | 0.9756 | 0.9756 | 2.84% | 83.88% |
| Random Forest | 0.8302 | 0.6048 | 0.9820 | 0.9819 | 2.35% | 87.05% |
| Gradient Boosted Trees | 0.8471 | 0.6199 | 0.9813 | 0.9812 | 2.40% | 86.34% |


---

👥 Contributors

Omkar Darekar – 22070122047

Manas Dhanpawde – 22070122108

Parimal Kulkarni – 22070122138

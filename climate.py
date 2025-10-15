# climate_change_prediction.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

class ClimateChangePredictor:
    def __init__(self):
        # Initialize Spark Session with proper Python configuration
        self.spark = SparkSession.builder \
            .appName("ClimateChangePrediction") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.pyspark.python", sys.executable) \
            .config("spark.pyspark.driver.python", sys.executable) \
            .getOrCreate()
        
        # Set log level to ERROR to reduce verbose output
        self.spark.sparkContext.setLogLevel("ERROR")
        
        self.df = None
        self.model = None
        self.predictions = None
        self.metrics = {}
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the climate dataset
        """
        print("Loading and preprocessing data...")
        
        # Define schema for the CSV file
        schema = StructType([
            StructField("STATION", StringType(), True),
            StructField("DATE", DateType(), True),
            StructField("LATITUDE", DoubleType(), True),
            StructField("LONGITUDE", DoubleType(), True),
            StructField("ELEVATION", DoubleType(), True),
            StructField("NAME", StringType(), True),
            StructField("PRCP", DoubleType(), True),
            StructField("PRCP_ATTRIBUTES", StringType(), True),
            StructField("TMAX", DoubleType(), True),
            StructField("TMAX_ATTRIBUTES", StringType(), True),
            StructField("TMIN", DoubleType(), True),
            StructField("TMIN_ATTRIBUTES", StringType(), True),
            StructField("TAVG", DoubleType(), True),
            StructField("TAVG_ATTRIBUTES", StringType(), True)
        ])
        
        # Load CSV data
        self.df = self.spark.read \
            .option("header", "true") \
            .option("ignoreLeadingWhiteSpace", "true") \
            .option("ignoreTrailingWhiteSpace", "true") \
            .schema(schema) \
            .csv(file_path)
        
        print(f"Original dataset count: {self.df.count()}")
        
        # Data preprocessing steps
        self._clean_data()
        self._feature_engineering()
        self._handle_missing_values()
        
        return self.df
    
    def _clean_data(self):
        """
        Clean the dataset - remove nulls and handle data quality issues
        """
        # Remove rows where critical columns are null
        self.df = self.df.filter(
            col("DATE").isNotNull() & 
            col("TAVG").isNotNull() &
            col("LATITUDE").isNotNull() &
            col("LONGITUDE").isNotNull()
        )
        
        # Convert temperature from tenths of degrees to actual degrees
        temperature_columns = ["TMAX", "TMIN", "TAVG"]
        for col_name in temperature_columns:
            self.df = self.df.withColumn(col_name, col(col_name) / 10.0)
        
        # Convert precipitation (assuming in tenths of mm)
        self.df = self.df.withColumn("PRCP", col("PRCP") / 10.0)
        
        print(f"After cleaning - dataset count: {self.df.count()}")
    
    def _feature_engineering(self):
        """
        Create new features for better prediction
        """
        # Extract temporal features
        self.df = self.df \
            .withColumn("YEAR", year("DATE")) \
            .withColumn("MONTH", month("DATE")) \
            .withColumn("DAY_OF_YEAR", dayofyear("DATE")) \
            .withColumn("SEASON", 
                       when((col("MONTH") >= 3) & (col("MONTH") <= 5), 1)  # Spring
                       .when((col("MONTH") >= 6) & (col("MONTH") <= 8), 2)  # Summer
                       .when((col("MONTH") >= 9) & (col("MONTH") <= 11), 3)  # Fall
                       .otherwise(0))  # Winter
        
        # Create temperature range
        self.df = self.df.withColumn("TEMP_RANGE", col("TMAX") - col("TMIN"))
        
        # Create decade feature for trend analysis
        self.df = self.df.withColumn("DECADE", (col("YEAR") / 10).cast("integer") * 10)
        
        # Create moving averages (using window functions)
        from pyspark.sql.window import Window
        window_spec = Window.orderBy("DATE").rowsBetween(-30, 0)
        self.df = self.df.withColumn("TAVG_30DAY_AVG", avg("TAVG").over(window_spec))
        
        print("Feature engineering completed")
    
    def _handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        # Fill missing precipitation with 0 (assuming no rain)
        self.df = self.df.fillna(0, subset=["PRCP"])
        
        # For missing TMAX/TMIN, use TAVG with reasonable offsets
        self.df = self.df.withColumn(
            "TMAX", 
            when(col("TMAX").isNull(), col("TAVG") + 5).otherwise(col("TMAX"))
        )
        
        self.df = self.df.withColumn(
            "TMIN", 
            when(col("TMIN").isNull(), col("TAVG") - 5).otherwise(col("TMIN"))
        )
        
        # Recalculate TEMP_RANGE after filling missing values
        self.df = self.df.withColumn("TEMP_RANGE", col("TMAX") - col("TMIN"))
        
        print("Missing values handled")
    
    def exploratory_data_analysis(self):
        """
        Perform exploratory data analysis and create visualizations
        """
        print("Performing Exploratory Data Analysis...")
        
        # Create visualizations directory
        os.makedirs("visualizations", exist_ok=True)
        
        # Get aggregated data for visualization
        yearly_avg = self.df.groupBy("YEAR").agg(avg("TAVG").alias("AVG_TAVG")).orderBy("YEAR").toPandas()
        monthly_avg = self.df.groupBy("MONTH").agg(avg("TAVG").alias("AVG_TAVG")).orderBy("MONTH").toPandas()
        
        # Sample data for distribution plots
        sample_df = self.df.sample(False, 0.1, seed=42).toPandas()
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Temperature trends over years
        plt.subplot(2, 3, 1)
        plt.plot(yearly_avg["YEAR"], yearly_avg["AVG_TAVG"], linewidth=2, color='#2E86AB')
        plt.title("Average Temperature Trend Over Years", fontsize=12, fontweight='bold')
        plt.xlabel("Year")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3)
        
        # 2. Seasonal patterns
        plt.subplot(2, 3, 2)
        colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        plt.bar(monthly_avg["MONTH"], monthly_avg["AVG_TAVG"], color=colors[0], alpha=0.7)
        plt.title("Monthly Average Temperature", fontsize=12, fontweight='bold')
        plt.xlabel("Month")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3, axis='y')
        
        # 3. Temperature distribution
        plt.subplot(2, 3, 3)
        plt.hist(sample_df["TAVG"], bins=50, alpha=0.7, color='#06A77D', edgecolor='black')
        plt.title("Temperature Distribution", fontsize=12, fontweight='bold')
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. Precipitation vs Temperature
        plt.subplot(2, 3, 4)
        plt.scatter(sample_df["PRCP"], sample_df["TAVG"], alpha=0.5, color='#D62828', s=10)
        plt.title("Precipitation vs Temperature", fontsize=12, fontweight='bold')
        plt.xlabel("Precipitation (mm)")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3)
        
        # 5. Temperature range by season
        plt.subplot(2, 3, 5)
        season_range = self.df.groupBy("SEASON").agg(avg("TEMP_RANGE").alias("AVG_RANGE")).orderBy("SEASON").toPandas()
        season_names = ['Winter', 'Spring', 'Summer', 'Fall']
        plt.bar(season_names, season_range["AVG_RANGE"], color=colors, alpha=0.7)
        plt.title("Average Temperature Range by Season", fontsize=12, fontweight='bold')
        plt.xlabel("Season")
        plt.ylabel("Temperature Range (°C)")
        plt.grid(True, alpha=0.3, axis='y')
        
        # 6. Decadal trends
        plt.subplot(2, 3, 6)
        decadal_avg = self.df.groupBy("DECADE").agg(avg("TAVG").alias("AVG_TAVG")).orderBy("DECADE").toPandas()
        plt.plot(decadal_avg["DECADE"], decadal_avg["AVG_TAVG"], marker='o', linewidth=2, 
                markersize=8, color='#F77F00')
        plt.title("Decadal Temperature Trend", fontsize=12, fontweight='bold')
        plt.xlabel("Decade")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("visualizations/eda_plots.png", dpi=300, bbox_inches='tight')
        print("✓ EDA plots saved")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total records: {self.df.count():,}")
        date_range = self.df.agg(min('YEAR'), max('YEAR')).collect()[0]
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        print(f"Number of unique stations: {self.df.select('STATION').distinct().count()}")
        
        # Temperature statistics
        temp_stats = self.df.select(
            mean("TAVG").alias("mean_temp"),
            stddev("TAVG").alias("std_temp"),
            min("TAVG").alias("min_temp"),
            max("TAVG").alias("max_temp")
        ).collect()[0]
        
        print(f"\nTemperature Statistics:")
        print(f"  Average Temperature: {temp_stats['mean_temp']:.2f}°C")
        print(f"  Standard Deviation: {temp_stats['std_temp']:.2f}°C")
        print(f"  Min Temperature: {temp_stats['min_temp']:.2f}°C")
        print(f"  Max Temperature: {temp_stats['max_temp']:.2f}°C")
        print("="*60 + "\n")
    
    def prepare_model_data(self):
        """
        Prepare data for machine learning model
        """
        print("Preparing data for machine learning...")
        
        # Select features for the model
        feature_columns = [
            "YEAR", "MONTH", "DAY_OF_YEAR", "LATITUDE", "LONGITUDE", 
            "ELEVATION", "PRCP", "TMAX", "TMIN", "TEMP_RANGE", "SEASON", "TAVG_30DAY_AVG"
        ]
        
        # Target variable
        target_column = "TAVG"
        
        # Remove any rows with null values in feature columns
        ml_df = self.df.select(feature_columns + [target_column]).dropna()
        
        # Split data into training and testing sets
        train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"Training set: {train_df.count():,} records")
        print(f"Test set: {test_df.count():,} records")
        
        return train_df, test_df, feature_columns, target_column
    
    def calculate_regression_metrics(self, predictions, label_col="TAVG"):
        """
        Calculate comprehensive regression metrics including classification-style metrics
        """
        # Convert to Pandas for easier calculation
        pred_pdf = predictions.select(label_col, "prediction").toPandas()
        y_true = pred_pdf[label_col].values
        y_pred = pred_pdf["prediction"].values
        
        # Calculate standard regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Adjusted R²
        n = len(y_true)
        p = 12  # number of features
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # ===== CLASSIFICATION-STYLE METRICS FOR REGRESSION =====
        # We'll convert continuous predictions to classes based on temperature ranges
        # This allows us to calculate Accuracy, Precision, Recall, and F1
        
        # Define temperature bins/classes (you can adjust these ranges)
        temp_bins = [-np.inf, 15, 20, 25, 30, 35, np.inf]
        temp_labels = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot', 'Very Hot']
        
        # Convert continuous values to categorical classes
        y_true_class = pd.cut(y_true, bins=temp_bins, labels=range(len(temp_labels)))
        y_pred_class = pd.cut(y_pred, bins=temp_bins, labels=range(len(temp_labels)))
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        
        # Alternative: Tolerance-based accuracy (within X degrees)
        tolerance = 2.0  # Consider prediction "correct" if within 2°C
        tolerance_accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance) * 100
        
        # Calculate per-class metrics for detailed analysis
        precision_per_class = precision_score(y_true_class, y_pred_class, 
                                             average=None, zero_division=0, 
                                             labels=range(len(temp_labels)))
        recall_per_class = recall_score(y_true_class, y_pred_class, 
                                       average=None, zero_division=0,
                                       labels=range(len(temp_labels)))
        f1_per_class = f1_score(y_true_class, y_pred_class, 
                               average=None, zero_division=0,
                               labels=range(len(temp_labels)))
        
        return {
            # Regression Metrics
            'RMSE': rmse,
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Adjusted_R2': adj_r2,
            'MAPE': mape,
            
            # Classification-style Metrics
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Tolerance_Accuracy': tolerance_accuracy,
            
            # Per-class metrics (for detailed reporting)
            'Precision_Per_Class': precision_per_class,
            'Recall_Per_Class': recall_per_class,
            'F1_Per_Class': f1_per_class,
            'Class_Labels': temp_labels
        }
    
    def train_model(self, train_df, test_df, feature_columns, target_column):
        """
        Train machine learning models for temperature prediction
        """
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # Create feature vector
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        
        # Standardize features
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", 
                               withStd=True, withMean=True)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(
                featuresCol="scaledFeatures", 
                labelCol=target_column,
                predictionCol="prediction",
                maxIter=100,
                regParam=0.01,
                elasticNetParam=0.5
            ),
            'Random Forest': RandomForestRegressor(
                featuresCol="scaledFeatures",
                labelCol=target_column,
                predictionCol="prediction",
                numTrees=100,
                maxDepth=10,
                seed=42
            ),
            'Gradient Boosted Trees': GBTRegressor(
                featuresCol="scaledFeatures",
                labelCol=target_column,
                predictionCol="prediction",
                maxIter=50,
                maxDepth=5,
                seed=42
            )
        }
        
        best_model = None
        best_metrics = None
        best_name = None
        best_r2 = -float('inf')
        
        all_results = []
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, model])
            
            # Train model
            trained_model = pipeline.fit(train_df)
            
            # Make predictions
            predictions = trained_model.transform(test_df)
            
            # Calculate metrics
            metrics = self.calculate_regression_metrics(predictions, target_column)
            
            # Store results
            all_results.append({
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R2'],
                'Adj R²': metrics['Adjusted_R2'],
                'MAPE': metrics['MAPE'],
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1_Score']
            })
            
            # Print metrics
            print(f"\n{'='*70}")
            print(f"{model_name} Performance Metrics")
            print(f"{'='*70}")
            
            print(f"\n📊 Regression Metrics:")
            print(f"  RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}°C")
            print(f"  MAE (Mean Absolute Error):      {metrics['MAE']:.4f}°C")
            print(f"  R² Score:                       {metrics['R2']:.4f}")
            print(f"  Adjusted R²:                    {metrics['Adjusted_R2']:.4f}")
            print(f"  MAPE (Mean Abs % Error):        {metrics['MAPE']:.2f}%")
            
            print(f"\n🎯 Classification Metrics (Temperature Categories):")
            print(f"  Accuracy:                       {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
            print(f"  Precision:                      {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
            print(f"  Recall:                         {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
            print(f"  F1-Score:                       {metrics['F1_Score']:.4f} ({metrics['F1_Score']*100:.2f}%)")
            print(f"  Tolerance Accuracy (±2°C):      {metrics['Tolerance_Accuracy']:.2f}%")
            
            # Print per-class metrics
            print(f"\n📈 Per-Category Performance:")
            print(f"  {'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print(f"  {'-'*51}")
            for i, label in enumerate(metrics['Class_Labels']):
                prec = metrics['Precision_Per_Class'][i]
                rec = metrics['Recall_Per_Class'][i]
                f1_c = metrics['F1_Per_Class'][i]
                print(f"  {label:<15} {prec:<12.4f} {rec:<12.4f} {f1_c:<12.4f}")
            print(f"{'='*70}\n")
            
            # Select best model based on R²
            if metrics['R2'] > best_r2:
                best_r2 = metrics['R2']
                best_model = trained_model
                best_metrics = metrics
                best_name = model_name
                self.predictions = predictions
        
        self.model = best_model
        self.metrics = best_metrics
        
        print("\n" + "="*60)
        print(f"✓ BEST MODEL: {best_name}")
        print("="*60)
        
        # Create comparison table
        results_df = pd.DataFrame(all_results)
        print("\nModel Comparison:")
        print(results_df.to_string(index=False))
        
        # Save metrics visualization
        self._plot_model_comparison(results_df)
        
        return self.model, self.predictions
    
    def _plot_model_comparison(self, results_df):
        """
        Create visualization comparing model performances
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Define metrics to plot
        regression_metrics = ['RMSE', 'MAE', 'R²', 'MAPE']
        classification_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        colors_reg = ['#E63946', '#F77F00', '#06A77D', '#4361EE']
        colors_clf = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        # Regression Metrics (Top row)
        for idx, (metric, color) in enumerate(zip(regression_metrics, colors_reg)):
            ax = fig.add_subplot(gs[0, idx % 3])
            if idx < 3:  # First 3 metrics in first row
                bars = ax.barh(results_df['Model'], results_df[metric], color=color, alpha=0.7)
                
                if metric == 'R²':
                    ax.axvline(x=0.95, color='green', linestyle='--', alpha=0.5, linewidth=2)
                    ax.text(0.95, len(results_df['Model'])-0.5, 'Excellent', 
                           fontsize=8, color='green', ha='left')
                
                ax.set_xlabel(metric, fontweight='bold', fontsize=10)
                ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=11)
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center', fontsize=9, 
                           fontweight='bold')
        
        # MAPE (special position)
        ax = fig.add_subplot(gs[1, 0])
        bars = ax.barh(results_df['Model'], results_df['MAPE'], color=colors_reg[3], alpha=0.7)
        ax.set_xlabel('MAPE (%)', fontweight='bold', fontsize=10)
        ax.set_title('MAPE Comparison', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Classification Metrics (Bottom two rows)
        positions = [(1, 1), (1, 2), (2, 0), (2, 1)]
        for idx, (metric, color, pos) in enumerate(zip(classification_metrics, colors_clf, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            bars = ax.barh(results_df['Model'], results_df[metric], color=color, alpha=0.7)
            
            # Add threshold lines
            ax.axvline(x=0.90, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(0.90, len(results_df['Model'])-0.5, 'Good', 
                   fontsize=7, color='green', ha='left')
            
            ax.set_xlabel(metric, fontweight='bold', fontsize=10)
            ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_xlim(0, 1.05)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Summary table
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for _, row in results_df.iterrows():
            summary_data.append([
                row['Model'][:15],  # Truncate long names
                f"{row['R²']:.3f}",
                f"{row['Accuracy']:.3f}",
                f"{row['F1-Score']:.3f}"
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Model', 'R²', 'Accuracy', 'F1'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the header
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best values
        best_r2_idx = results_df['R²'].idxmax() + 1
        best_acc_idx = results_df['Accuracy'].idxmax() + 1
        best_f1_idx = results_df['F1-Score'].idxmax() + 1
        
        table[(best_r2_idx, 1)].set_facecolor('#90EE90')
        table[(best_acc_idx, 2)].set_facecolor('#90EE90')
        table[(best_f1_idx, 3)].set_facecolor('#90EE90')
        
        ax.set_title('Performance Summary', fontweight='bold', fontsize=11, pad=20)
        
        plt.suptitle('Comprehensive Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig("visualizations/model_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Model comparison plot saved")
    
    def predict_future_temperatures(self, future_years=10):
        """
        Predict temperatures for future years
        """
        print(f"\nPredicting temperatures for next {future_years} years...")
        
        # Get the latest date in the dataset
        max_year = self.df.agg(max("YEAR")).collect()[0][0]
        
        # Get recent statistics
        recent_stats = self.df.filter(col("YEAR") >= max_year - 5).groupBy().agg(
            avg("LATITUDE").alias("lat"),
            avg("LONGITUDE").alias("lon"),
            avg("ELEVATION").alias("elev"),
            avg("PRCP").alias("prcp"),
            avg("TAVG").alias("tavg_baseline")
        ).collect()[0]
        
        future_data = []
        current_year = max_year + 1
        
        for year in range(current_year, current_year + future_years):
            for month in range(1, 13):
                # Calculate day of year for mid-month
                day_of_year = (datetime(year, month, 15) - datetime(year, 1, 1)).days + 1
                
                # Get historical monthly statistics
                monthly_stats = self.df.filter(
                    (col("MONTH") == month) & (col("YEAR") >= max_year - 10)
                ).groupBy().agg(
                    avg("TMAX").alias("avg_tmax"),
                    avg("TMIN").alias("avg_tmin"),
                    avg("TAVG").alias("avg_tavg")
                ).collect()[0]
                
                # Determine season
                if 3 <= month <= 5:
                    season = 1
                elif 6 <= month <= 8:
                    season = 2
                elif 9 <= month <= 11:
                    season = 3
                else:
                    season = 0
                
                future_data.append({
                    "YEAR": year,
                    "MONTH": month,
                    "DAY_OF_YEAR": day_of_year,
                    "LATITUDE": recent_stats["lat"],
                    "LONGITUDE": recent_stats["lon"],
                    "ELEVATION": recent_stats["elev"],
                    "PRCP": recent_stats["prcp"],
                    "TMAX": monthly_stats["avg_tmax"],
                    "TMIN": monthly_stats["avg_tmin"],
                    "TEMP_RANGE": monthly_stats["avg_tmax"] - monthly_stats["avg_tmin"],
                    "SEASON": season,
                    "TAVG_30DAY_AVG": monthly_stats["avg_tavg"]
                })
        
        # Create future DataFrame
        future_df = self.spark.createDataFrame(future_data)
        
        # Make predictions
        future_predictions = self.model.transform(future_df)
        
        # Analyze future trends
        future_trends = future_predictions.groupBy("YEAR").agg(
            avg("prediction").alias("predicted_temp")
        ).orderBy("YEAR").toPandas()
        
        # Get historical trends
        historical_trends = self.df.groupBy("YEAR").agg(
            avg("TAVG").alias("historical_temp")
        ).orderBy("YEAR").toPandas()
        
        # Create prediction visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Historical vs Predicted
        ax1.plot(historical_trends["YEAR"], historical_trends["historical_temp"], 
                label="Historical", marker='o', linewidth=2, color='#2E86AB', markersize=3)
        ax1.plot(future_trends["YEAR"], future_trends["predicted_temp"], 
                label="Predicted", marker='s', linewidth=2, color='#E63946', markersize=6)
        ax1.axvline(x=max_year, color='gray', linestyle='--', alpha=0.7, label='Prediction Start')
        ax1.set_title("Temperature Trends: Historical vs Predicted", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Year", fontweight='bold')
        ax1.set_ylabel("Average Temperature (°C)", fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temperature change
        start_temp = future_trends["predicted_temp"].iloc[0]
        temp_changes = [(temp - start_temp) for temp in future_trends["predicted_temp"]]
        
        colors_bar = ['#06A77D' if x >= 0 else '#E63946' for x in temp_changes]
        ax2.bar(future_trends["YEAR"], temp_changes, color=colors_bar, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title("Predicted Temperature Change from Baseline", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Year", fontweight='bold')
        ax2.set_ylabel("Temperature Change (°C)", fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("visualizations/future_predictions.png", dpi=300, bbox_inches='tight')
        print("✓ Future predictions plot saved")
        
        print("\n" + "="*60)
        print("FUTURE TEMPERATURE PREDICTIONS")
        print("="*60)
        for _, row in future_trends.iterrows():
            print(f"Year {int(row['YEAR'])}: {row['predicted_temp']:.2f}°C")
        print("="*60 + "\n")
        
        return future_predictions
    
    def analyze_climate_impact(self):
        """
        Analyze potential climate change impacts
        """
        print("Analyzing climate change impacts...")
        
        # Calculate temperature increase rate
        temp_trend = self.df.filter(col("YEAR") >= 1950).groupBy("YEAR").agg(
            avg("TAVG").alias("avg_temp")
        ).orderBy("YEAR").toPandas()
        
        if len(temp_trend) > 1:
            # Linear regression for trend
            x = temp_trend["YEAR"].values
            y = temp_trend["avg_temp"].values
            slope, intercept = np.polyfit(x, y, 1)
            
            print("\n" + "="*60)
            print("CLIMATE IMPACT ANALYSIS")
            print("="*60)
            print(f"Historical temperature trend: {slope:.4f}°C per year")
            print(f"Projected increase in 10 years: {slope * 10:.2f}°C")
            print(f"Projected increase in 50 years: {slope * 50:.2f}°C")
            
            # Impact assessment
            if slope > 0:
                print("\n📈 Trend: WARMING pattern detected")
                if slope > 0.01:
                    print("⚠️  Significant warming trend - Potential climate change impact")
                else:
                    print("ℹ️  Moderate warming trend")
            else:
                print("\n📉 Trend: COOLING pattern detected")
            print("="*60)
        
        # Create impact visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Temperature trend with trendline
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(temp_trend["YEAR"], temp_trend["avg_temp"], marker='o', 
                linewidth=2, markersize=4, label='Observed', color='#2E86AB')
        z = np.polyfit(temp_trend["YEAR"], temp_trend["avg_temp"], 1)
        p = np.poly1d(z)
        ax1.plot(temp_trend["YEAR"], p(temp_trend["YEAR"]), "r--", 
                alpha=0.8, linewidth=2, label=f'Trend (slope={slope:.4f}°C/year)')
        ax1.set_title("Temperature Trend with Linear Fit", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Year", fontweight='bold')
        ax1.set_ylabel("Temperature (°C)", fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Seasonal patterns
        ax2 = fig.add_subplot(gs[1, 0])
        seasonal_data = self.df.groupBy("YEAR", "SEASON").agg(
            avg("TAVG").alias("seasonal_temp")
        ).orderBy("YEAR").toPandas()
        
        season_names = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
        season_colors = {0: '#4361EE', 1: '#06A77D', 2: '#F77F00', 3: '#E63946'}
        
        for season_num, season_name in season_names.items():
            season_subset = seasonal_data[seasonal_data["SEASON"] == season_num]
            if len(season_subset) > 0:
                ax2.plot(season_subset["YEAR"], season_subset["seasonal_temp"], 
                        label=season_name, marker='o', markersize=2, linewidth=1.5,
                        color=season_colors[season_num])
        
        ax2.set_title("Seasonal Temperature Trends", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Year", fontweight='bold')
        ax2.set_ylabel("Temperature (°C)", fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temperature distribution changes
        ax3 = fig.add_subplot(gs[1, 1])
        early_period = self.df.filter(col("YEAR") <= 1980).select("TAVG").toPandas()
        late_period = self.df.filter(col("YEAR") >= 2000).select("TAVG").toPandas()
        
        if len(early_period) > 0 and len(late_period) > 0:
            ax3.hist(early_period["TAVG"], bins=40, alpha=0.6, label="Early Period (≤1980)", 
                    density=True, color='#4361EE', edgecolor='black')
            ax3.hist(late_period["TAVG"], bins=40, alpha=0.6, label="Late Period (≥2000)", 
                    density=True, color='#E63946', edgecolor='black')
            ax3.set_title("Temperature Distribution Shift", fontsize=12, fontweight='bold')
            ax3.set_xlabel("Temperature (°C)", fontweight='bold')
            ax3.set_ylabel("Density", fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Extreme temperature events
        ax4 = fig.add_subplot(gs[2, 0])
        extreme_heat = self.df.groupBy("YEAR").agg(
            max("TMAX").alias("max_temp")
        ).orderBy("YEAR").toPandas()
        
        ax4.plot(extreme_heat["YEAR"], extreme_heat["max_temp"], 
                marker='o', linewidth=2, markersize=3, color='#E63946')
        ax4.set_title("Annual Maximum Temperature Trend", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Year", fontweight='bold')
        ax4.set_ylabel("Max Temperature (°C)", fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Precipitation trends
        ax5 = fig.add_subplot(gs[2, 1])
        precip_trend = self.df.groupBy("YEAR").agg(
            sum("PRCP").alias("annual_precip")
        ).orderBy("YEAR").toPandas()
        
        ax5.bar(precip_trend["YEAR"], precip_trend["annual_precip"], 
               alpha=0.7, color='#06A77D', edgecolor='black', width=0.8)
        ax5.set_title("Annual Precipitation Trend", fontsize=12, fontweight='bold')
        ax5.set_xlabel("Year", fontweight='bold')
        ax5.set_ylabel("Total Precipitation (mm)", fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle("Climate Impact Analysis", fontsize=16, fontweight='bold', y=0.995)
        plt.savefig("visualizations/climate_impact_analysis.png", dpi=300, bbox_inches='tight')
        print("✓ Climate impact analysis plot saved")
    
    def generate_report(self):
        """
        Generate a comprehensive report of the analysis
        """
        print("\n" + "="*80)
        print(" " * 20 + "CLIMATE CHANGE PREDICTION REPORT")
        print("="*80)
        
        # Basic statistics
        total_records = self.df.count()
        date_range = self.df.agg(min("YEAR"), max("YEAR")).collect()[0]
        stations_count = self.df.select("STATION").distinct().count()
        
        print(f"\n{'Dataset Overview:':<30}")
        print(f"  {'Total records:':<28} {total_records:>15,}")
        print(f"  {'Date range:':<28} {date_range[0]:>8} to {date_range[1]}")
        print(f"  {'Weather stations:':<28} {stations_count:>15}")
        
        # Temperature statistics
        temp_stats = self.df.select(
            mean("TAVG").alias("mean"),
            stddev("TAVG").alias("std"),
            min("TAVG").alias("min"),
            max("TAVG").alias("max")
        ).collect()[0]
        
        print(f"\n{'Temperature Statistics:':<30}")
        print(f"  {'Average temperature:':<28} {temp_stats['mean']:>14.2f}°C")
        print(f"  {'Standard deviation:':<28} {temp_stats['std']:>14.2f}°C")
        print(f"  {'Min temperature:':<28} {temp_stats['min']:>14.2f}°C")
        print(f"  {'Max temperature:':<28} {temp_stats['max']:>14.2f}°C")
        
        # Model performance
        if self.metrics:
            print(f"\n{'Best Model Performance Metrics:':<30}")
            print(f"\n{'  REGRESSION METRICS:':<30}")
            print(f"  {'RMSE (Root Mean Squared Error):':<35} {self.metrics['RMSE']:>14.4f}°C")
            print(f"  {'MAE (Mean Absolute Error):':<35} {self.metrics['MAE']:>14.4f}°C")
            print(f"  {'R² Score:':<35} {self.metrics['R2']:>14.4f}")
            print(f"  {'Adjusted R²:':<35} {self.metrics['Adjusted_R2']:>14.4f}")
            print(f"  {'MAPE (Mean Abs % Error):':<35} {self.metrics['MAPE']:>13.2f}%")
            
            print(f"\n{'  CLASSIFICATION METRICS:':<30}")
            print(f"  {'Accuracy:':<35} {self.metrics['Accuracy']:>14.4f} ({self.metrics['Accuracy']*100:.2f}%)")
            print(f"  {'Precision:':<35} {self.metrics['Precision']:>14.4f} ({self.metrics['Precision']*100:.2f}%)")
            print(f"  {'Recall:':<35} {self.metrics['Recall']:>14.4f} ({self.metrics['Recall']*100:.2f}%)")
            print(f"  {'F1-Score:':<35} {self.metrics['F1_Score']:>14.4f} ({self.metrics['F1_Score']*100:.2f}%)")
            print(f"  {'Tolerance Accuracy (±2°C):':<35} {self.metrics['Tolerance_Accuracy']:>13.2f}%")
            
            # Interpretation
            print(f"\n{'Model Quality Assessment:':<30}")
            if self.metrics['R2'] > 0.95:
                print("  ✓ Excellent regression performance (R² > 0.95)")
            elif self.metrics['R2'] > 0.90:
                print("  ✓ Very good regression performance (R² > 0.90)")
            elif self.metrics['R2'] > 0.80:
                print("  ✓ Good regression performance (R² > 0.80)")
            else:
                print("  ⚠ Moderate regression performance")
            
            if self.metrics['Accuracy'] > 0.90:
                print("  ✓ Excellent classification accuracy (>90%)")
            elif self.metrics['Accuracy'] > 0.80:
                print("  ✓ Good classification accuracy (>80%)")
            else:
                print("  ⚠ Moderate classification accuracy")
            
            if self.metrics['F1_Score'] > 0.90:
                print("  ✓ Excellent F1-Score (>0.90)")
            elif self.metrics['F1_Score'] > 0.80:
                print("  ✓ Good F1-Score (>0.80)")
            else:
                print("  ⚠ Moderate F1-Score")
            
            if self.metrics['MAPE'] < 5:
                print("  ✓ High prediction accuracy (MAPE < 5%)")
            elif self.metrics['MAPE'] < 10:
                print("  ✓ Good prediction accuracy (MAPE < 10%)")
            else:
                print("  ⚠ Acceptable prediction accuracy")
        
        print(f"\n{'Output Files Generated:':<30}")
        print(f"  • EDA Plots: visualizations/eda_plots.png")
        print(f"  • Model Comparison: visualizations/model_comparison.png")
        print(f"  • Future Predictions: visualizations/future_predictions.png")
        print(f"  • Climate Impact Analysis: visualizations/climate_impact_analysis.png")
        
        # Additional insights
        recent_trend = self.df.filter(col("YEAR") >= 2000).groupBy("YEAR").agg(
            avg("TAVG").alias("avg_temp")
        ).orderBy("YEAR").toPandas()
        
        if len(recent_trend) > 1:
            slope = np.polyfit(recent_trend["YEAR"], recent_trend["avg_temp"], 1)[0]
            print(f"\n{'Recent Climate Trends (2000+):':<30}")
            print(f"  {'Temperature change rate:':<28} {slope:>13.4f}°C/year")
            print(f"  {'10-year projection:':<28} {slope*10:>13.2f}°C increase")
        
        print("\n" + "="*80)
        print(" " * 25 + "✓ Analysis Completed Successfully!")
        print("="*80 + "\n")
    
    def run_complete_analysis(self, file_path):
        """
        Run the complete climate change analysis pipeline
        """
        print("\n" + "="*80)
        print(" " * 15 + "STARTING COMPLETE CLIMATE CHANGE ANALYSIS")
        print("="*80 + "\n")
        
        try:
            # 1. Load and preprocess data
            self.load_and_preprocess_data(file_path)
            
            # 2. Exploratory Data Analysis
            self.exploratory_data_analysis()
            
            # 3. Prepare ML data
            train_df, test_df, feature_columns, target_column = self.prepare_model_data()
            
            # 4. Train models
            self.train_model(train_df, test_df, feature_columns, target_column)
            
            # 5. Predict future temperatures
            self.predict_future_temperatures(future_years=10)
            
            # 6. Analyze climate impact
            self.analyze_climate_impact()
            
            # 7. Generate final report
            self.generate_report()
            
            print("✅ All visualizations have been saved to the 'visualizations' folder")
            print("✅ Analysis completed successfully!\n")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def stop_spark(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            print("Spark session stopped")

def main():
    """
    Main function to run the climate change prediction project
    """
    # Initialize the predictor
    predictor = ClimateChangePredictor()
    
    try:
        # File path - update this to your CSV file location
        file_path = "AE000041196.csv"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ Error: File '{file_path}' not found!")
            print("Please ensure the CSV file is in the current directory.")
            return
        
        # Run complete analysis
        predictor.run_complete_analysis(file_path)
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        predictor.stop_spark()

if __name__ == "__main__":
    main()
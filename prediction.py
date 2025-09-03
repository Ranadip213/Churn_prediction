from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# -------------------------
# Step 0: Start Spark session
# -------------------------
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# -------------------------
# Step 1: Load saved pipeline
# -------------------------
model_path = "/content/model/churn_pipeline"
pipeline_model = PipelineModel.load(model_path)
print("✅ Pipeline loaded successfully")

# -------------------------
# Step 2: Load new dataset
# -------------------------
input_csv = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = spark.read.csv(input_csv, header=True, inferSchema=True)

# -------------------------
# Step 3: Preprocess + Predict
# -------------------------
predictions = pipeline_model.transform(df)

# -------------------------
# Step 4: Show results
# -------------------------
predictions.select("customerID", "probability", "prediction").show(10, truncate=False)

# -------------------------
# Step 5: Save predictions
# -------------------------
predictions.select("customerID", "prediction") \
           .toPandas() \
           .to_csv("predictions.csv", index=False)

print("✅ Predictions saved to predictions.csv")

# -------------------------
# Step 6: Stop Spark session
# -------------------------
spark.stop()

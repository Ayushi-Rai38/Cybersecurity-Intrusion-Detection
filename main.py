"""
Cybersecurity Intrusion Detection System
=========================================
PySpark MLlib pipeline: data loading, cleaning, feature engineering,
Random Forest classification, evaluation, and prediction export.

Dataset  : UNSW-NB15
Tasks    : 367-372 (all six required tasks)
Spark    : 4.2.0  (PySpark pip package 4.0.0)
Python   : 3.12.x
Java     : JDK 8  (JAVA_HOME = C:/Program Files/Java/jdk-1.8)
Hadoop   : 3.2.4  (HADOOP_HOME = C:/hadoopsetup/hadoop-3.2.4 — winutils)

Run via the project virtual environment:
    .venv/Scripts/python main.py
"""

import os
import sys
import json
import time
import logging

# ─── Point PySpark at the existing Spark/Java installation ────────────────────
# These are already set as system env-vars; we reinforce them here so the
# script works correctly regardless of which terminal activates the .venv.
os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")
os.environ.setdefault("SPARK_HOME",   r"C:\spark-4.2.0-bin-hadoop3")
os.environ.setdefault("HADOOP_HOME",  r"C:\hadoopsetup\hadoop-3.2.4")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)   # use venv Python
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Path constants ───────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(BASE_DIR, "data", "UNSW_NB15_training-set.csv")
TEST_CSV   = os.path.join(BASE_DIR, "data", "UNSW_NB15_testing-set.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "intrusion_rf_model")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
METRICS_F  = os.path.join(OUT_DIR, "metrics.json")
PREDS_CSV  = os.path.join(OUT_DIR, "predictions.csv")
CM_CSV     = os.path.join(OUT_DIR, "confusion_matrix.csv")

# ─── Guard: dataset files ─────────────────────────────────────────────────────
for path, name in [(TRAIN_CSV, "Training CSV"), (TEST_CSV, "Testing CSV")]:
    if not os.path.exists(path):
        log.error("MISSING FILE: %s  expected at  %s", name, path)
        sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)

# ─── PySpark imports ──────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StringType, LongType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        StringIndexer, OneHotEncoder, VectorAssembler, Imputer
    )
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import (
        BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    )
    from pyspark.ml import PipelineModel
except ImportError as exc:
    log.error("PySpark import failed: %s", exc)
    log.error("Run:  pip install pyspark")
    sys.exit(1)

# ─── Spark session ────────────────────────────────────────────────────────────
log.info("Starting Spark session …")
try:
    spark = (
        SparkSession.builder
        .appName("CybersecurityIntrusionDetection")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.network.timeout", "600s")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        # Spark 4.x: suppress excessive Hadoop/YARN warnings on Windows
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log.info("Spark %s started.", spark.version)
except Exception as exc:
    log.error("Cannot start Spark: %s", exc)
    log.error("Ensure Java 8/11/17 is installed and JAVA_HOME is set.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 367 — Load and clean data
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 367: Loading and cleaning data")
log.info("=" * 60)

def load_csv(path: str, label: str):
    """Load a CSV into a Spark DataFrame with header and schema inference."""
    log.info("Loading %s …", label)
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "false")
        .csv(path)
    )
    # Strip BOM from first column name if present
    renamed = {}
    for col in df.columns:
        clean = col.lstrip("\ufeff").strip()
        if clean != col:
            renamed[col] = clean
    for old, new in renamed.items():
        df = df.withColumnRenamed(old, new)
    log.info("  %s: %d rows × %d columns", label, df.count(), len(df.columns))
    return df

train_raw = load_csv(TRAIN_CSV, "Training set")
test_raw  = load_csv(TEST_CSV,  "Testing set")

# ─── Schema and sample ───────────────────────────────────────────────────────
log.info("\n── Training Schema ──")
train_raw.printSchema()
log.info("\n── Training Sample (3 rows) ──")
train_raw.show(3, truncate=False)

# ─── Data quality report ──────────────────────────────────────────────────────
log.info("\n── Data Quality Report ──")

def quality_report(df, name):
    total = df.count()
    log.info("%s: %d rows, %d columns", name, total, len(df.columns))

    # Null / empty counts per column
    null_exprs = [
        F.sum(
            F.when(F.col(c).isNull() | (F.col(c).cast("string") == ""), 1).otherwise(0)
        ).alias(c)
        for c in df.columns
    ]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    bad_cols = {k: v for k, v in null_counts.items() if v > 0}
    if bad_cols:
        log.info("  Columns with nulls/blanks: %s", bad_cols)
    else:
        log.info("  No null or blank values found.")

    # Duplicate rows
    dupes = total - df.dropDuplicates().count()
    log.info("  Duplicate rows: %d", dupes)

    # Label distribution
    log.info("  Label distribution:")
    df.groupBy("label").count().orderBy("label").show()

quality_report(train_raw, "Training")
quality_report(test_raw,  "Testing")

# ─── Numeric columns (everything except categorical & target) ─────────────────
CATEGORICAL_COLS = ["proto", "service", "state"]
ID_COLS          = ["id"]
TARGET_COL       = "label"
LEAKY_COLS       = ["attack_cat"]   # kept for dashboard, excluded from model

# Discover numeric columns
ALL_NUMERIC = [
    c for c in train_raw.columns
    if c not in CATEGORICAL_COLS + ID_COLS + [TARGET_COL] + LEAKY_COLS
]
log.info("Numeric feature columns (%d): %s", len(ALL_NUMERIC), ALL_NUMERIC)

# ─── Cast all numeric columns to Double ──────────────────────────────────────
def cast_numeric(df, cols):
    for c in cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
    return df

train_clean = cast_numeric(train_raw, ALL_NUMERIC)
test_clean  = cast_numeric(test_raw,  ALL_NUMERIC)

# ─── Replace Infinity with null (Spark keeps inf from inference) ──────────────
def replace_inf(df, cols):
    for c in cols:
        df = df.withColumn(
            c,
            F.when(F.col(c).isin([float("inf"), float("-inf")]), None)
             .otherwise(F.col(c))
        )
    return df

train_clean = replace_inf(train_clean, ALL_NUMERIC)
test_clean  = replace_inf(test_clean,  ALL_NUMERIC)

# ─── Fill missing categoricals with "Unknown" ────────────────────────────────
for c in CATEGORICAL_COLS:
    train_clean = train_clean.withColumn(
        c, F.when(F.col(c).isNull() | (F.col(c) == ""), "Unknown").otherwise(F.col(c))
    )
    test_clean = test_clean.withColumn(
        c, F.when(F.col(c).isNull() | (F.col(c) == ""), "Unknown").otherwise(F.col(c))
    )

# ─── Cast target label to Integer ────────────────────────────────────────────
train_clean = train_clean.withColumn(TARGET_COL, F.col(TARGET_COL).cast(IntegerType()))
test_clean  = test_clean.withColumn(TARGET_COL,  F.col(TARGET_COL).cast(IntegerType()))

# ─── Guard: both classes must exist ──────────────────────────────────────────
label_vals = [r[0] for r in train_clean.select(TARGET_COL).distinct().collect()]
log.info("Distinct label values in training: %s", sorted(label_vals))
if 0 not in label_vals or 1 not in label_vals:
    log.error("Training data does not contain both label classes (0 and 1). Aborting.")
    spark.stop()
    sys.exit(1)

log.info("TASK 367 complete.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 368 — Feature Engineering (MLlib Pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 368: Feature engineering")
log.info("=" * 60)

# ─── Feature selection ────────────────────────────────────────────────────────
# Core traffic features from UNSW-NB15 (no sport/dsport in this dataset)
NUMERIC_FEATURES = [
    "dur",           # connection duration
    "sbytes",        # source bytes
    "dbytes",        # destination bytes
    "spkts",         # source packet count
    "dpkts",         # destination packet count
    "rate",          # packet rate
    "sttl",          # source time-to-live
    "dttl",          # destination time-to-live
    "sload",         # source bits/second load
    "dload",         # destination bits/second load
    "sloss",         # source packet loss
    "dloss",         # destination packet loss
    "sinpkt",        # source inter-packet arrival (ms)
    "dinpkt",        # destination inter-packet arrival (ms)
    "sjit",          # source jitter
    "djit",          # destination jitter
    "swin",          # source TCP window size
    "dwin",          # destination TCP window size
    "tcprtt",        # TCP round-trip time
    "synack",        # SYN→ACK time
    "ackdat",        # ACK→data time
    "smean",         # mean source packet size
    "dmean",         # mean destination packet size
    "trans_depth",   # HTTP request/response depth
    "response_body_len",
    "ct_srv_src",    # connections to same service from same src (last 100)
    "ct_state_ttl",  # state/TTL combination counter
    "ct_dst_ltm",    # connections to same dst (last 100)
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",  # FTP login indicator (NOT a general failed-login count)
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "stcpb",         # source TCP base sequence number
    "dtcpb",         # destination TCP base sequence number
]

# Keep only those that actually exist in the dataset
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in train_clean.columns]
log.info("Numeric features selected (%d): %s", len(NUMERIC_FEATURES), NUMERIC_FEATURES)

# NOTE: is_ftp_login is an FTP-specific login indicator, NOT a general
# "failed login count". It is used here only as a binary FTP-activity flag.
# The UNSW-NB15 dataset does not include a general failed-login-attempt counter.

# ─── Pipeline stages ──────────────────────────────────────────────────────────

# Stage 1: Imputer for numeric nulls (fit on training only)
imputer = Imputer(
    inputCols=NUMERIC_FEATURES,
    outputCols=[f"{c}_imp" for c in NUMERIC_FEATURES],
    strategy="median"
)

IMPUTED_FEATURES = [f"{c}_imp" for c in NUMERIC_FEATURES]

# Stage 2: StringIndexer for each categorical column
cat_indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"   # unseen categories → extra index
    )
    for c in CATEGORICAL_COLS
]

# Stage 3: OneHotEncoder for indexed categoricals
cat_ohe_inputs  = [f"{c}_idx" for c in CATEGORICAL_COLS]
cat_ohe_outputs = [f"{c}_ohe" for c in CATEGORICAL_COLS]
ohe = OneHotEncoder(
    inputCols=cat_ohe_inputs,
    outputCols=cat_ohe_outputs,
    handleInvalid="keep"
)

# Stage 4: VectorAssembler
assembler_inputs = IMPUTED_FEATURES + cat_ohe_outputs
assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features",
    handleInvalid="keep"
)

log.info("TASK 368 complete.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 369 — Random Forest Classifier
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 369: Training Random Forest classifier")
log.info("=" * 60)

# ─── Class imbalance: compute class weights ──────────────────────────────────
# UNSW-NB15 training: ~56 k normal, ~119 k attack → attack is majority
# We weight the minority (normal / class 0) more heavily.
label_counts = (
    train_clean.groupBy(TARGET_COL).count().collect()
)
count_map = {r[TARGET_COL]: r["count"] for r in label_counts}
total_train = sum(count_map.values())
n_classes   = len(count_map)
log.info("Training class counts: %s", count_map)

# weightCol approach: weight = total / (n_classes * class_count)
def class_weight(label):
    return total_train / (n_classes * count_map[label])

w0 = class_weight(0)
w1 = class_weight(1)
log.info("Class weights  →  normal (0): %.4f   attack (1): %.4f", w0, w1)

# Add weight column to training data
train_weighted = train_clean.withColumn(
    "classWeight",
    F.when(F.col(TARGET_COL) == 0, w0).otherwise(w1)
)

# ─── Build pipeline ──────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol=TARGET_COL,
    weightCol="classWeight",
    numTrees=100,
    maxDepth=10,
    seed=42,
    predictionCol="prediction",
    rawPredictionCol="rawPrediction",
    probabilityCol="probability",
)

pipeline = Pipeline(stages=cat_indexers + [ohe, imputer, assembler, rf])

log.info("Fitting pipeline on training data …")
t0 = time.time()
try:
    pipeline_model = pipeline.fit(train_weighted)
except Exception as exc:
    log.error("Pipeline training failed: %s", exc)
    spark.stop()
    sys.exit(1)
elapsed = time.time() - t0
log.info("Training complete in %.1f seconds.", elapsed)

# ─── Save model ───────────────────────────────────────────────────────────────
log.info("Saving pipeline model to %s …", MODEL_DIR)
try:
    pipeline_model.write().overwrite().save(MODEL_DIR)
    log.info("Model saved.")
except Exception as exc:
    log.warning("Could not save model: %s", exc)

log.info("TASK 369 complete.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 370 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 370: Evaluating model on test set")
log.info("=" * 60)

# Apply pipeline to test data (no weight column needed for inference)
log.info("Generating predictions on test set …")
predictions = pipeline_model.transform(test_clean)

# ─── Evaluators ───────────────────────────────────────────────────────────────
roc_eval = BinaryClassificationEvaluator(
    labelCol=TARGET_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
acc_eval = MulticlassClassificationEvaluator(
    labelCol=TARGET_COL,
    predictionCol="prediction",
    metricName="accuracy"
)

roc_auc  = roc_eval.evaluate(predictions)
accuracy = acc_eval.evaluate(predictions)
log.info("ROC-AUC:  %.4f", roc_auc)
log.info("Accuracy: %.4f", accuracy)

# ─── Attack-class (label = 1) precision / recall / F1 ────────────────────────
# MulticlassClassificationEvaluator with weightedPrecision etc. doesn't give
# per-class metrics cleanly; we compute manually from the confusion matrix.

# Confusion matrix via Spark SQL
cm = (
    predictions
    .groupBy(TARGET_COL, "prediction")
    .count()
    .collect()
)

tn = fp = fn = tp = 0
for row in cm:
    actual = int(row[TARGET_COL])
    pred   = int(row["prediction"])
    cnt    = row["count"]
    if actual == 0 and pred == 0:
        tn = cnt
    elif actual == 0 and pred == 1:
        fp = cnt
    elif actual == 1 and pred == 0:
        fn = cnt
    elif actual == 1 and pred == 1:
        tp = cnt

precision_attack = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_attack    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_attack = (
    2 * precision_attack * recall_attack / (precision_attack + recall_attack)
    if (precision_attack + recall_attack) > 0 else 0.0
)

log.info("\n── Confusion Matrix ──")
log.info("                  Predicted 0   Predicted 1")
log.info("Actual 0 (Normal) %10d    %10d", tn, fp)
log.info("Actual 1 (Attack) %10d    %10d", fn, tp)

log.info("\n── Attack-Class Metrics (label = 1) ──")
log.info("  Precision: %.4f", precision_attack)
log.info("  Recall:    %.4f", recall_attack)
log.info("  F1-score:  %.4f", f1_attack)
log.info("  ROC-AUC:   %.4f", roc_auc)
log.info("  Accuracy:  %.4f", accuracy)

# ─── Save metrics ────────────────────────────────────────────────────────────
metrics = {
    "accuracy":          round(accuracy, 6),
    "attack_precision":  round(precision_attack, 6),
    "attack_recall":     round(recall_attack, 6),
    "attack_f1":         round(f1_attack, 6),
    "roc_auc":           round(roc_auc, 6),
    "confusion_matrix": {
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp)
    },
    "class_distribution": {
        "training_normal": int(count_map.get(0, 0)),
        "training_attack": int(count_map.get(1, 0)),
    }
}
with open(METRICS_F, "w") as fh:
    json.dump(metrics, fh, indent=2)
log.info("Metrics saved → %s", METRICS_F)

# ─── Save confusion matrix CSV ───────────────────────────────────────────────
import pandas as pd
cm_df = pd.DataFrame(
    {"Predicted_Normal": [tn, fn], "Predicted_Attack": [fp, tp]},
    index=["Actual_Normal", "Actual_Attack"]
)
cm_df.to_csv(CM_CSV)
log.info("Confusion matrix saved → %s", CM_CSV)

log.info("TASK 370 complete.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 371 — Test Predictions and Intrusion Risk
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 371: Generating intrusion-risk predictions")
log.info("=" * 60)

# Extract attack probability (index 1 of probability vector)
extract_prob = F.udf(lambda v: float(v[1]) if v is not None else 0.0, DoubleType())

# Configurable risk thresholds
HIGH_THRESH   = 0.80
MEDIUM_THRESH = 0.50

pred_df = (
    predictions
    .withColumn("attack_prob", extract_prob(F.col("probability")))
    .withColumn(
        "predicted_status",
        F.when(F.col("prediction") == 1, "Suspicious").otherwise("Benign")
    )
    .withColumn(
        "actual_status",
        F.when(F.col(TARGET_COL) == 1, "Attack").otherwise("Normal")
    )
    .withColumn(
        "risk_level",
        F.when(F.col("attack_prob") >= HIGH_THRESH, "High")
         .when(F.col("attack_prob") >= MEDIUM_THRESH, "Medium")
         .otherwise("Low")
    )
)

# Select output columns
output_cols = [
    TARGET_COL,         # actual label (0/1)
    "prediction",       # predicted label (0/1)
    "actual_status",
    "predicted_status",
    "attack_prob",
    "risk_level",
    "attack_cat",
    "proto",
    "service",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "rate",
    "sttl",
    "dttl",
]
# Keep only columns that exist
output_cols = [c for c in output_cols if c in pred_df.columns]

log.info("Converting predictions to Pandas for CSV export …")
pred_pandas = pred_df.select(output_cols).toPandas()
pred_pandas.to_csv(PREDS_CSV, index=False)
log.info("Predictions saved → %s  (%d rows)", PREDS_CSV, len(pred_pandas))

# ─── Risk level summary ───────────────────────────────────────────────────────
log.info("\n── Risk Level Distribution ──")
pred_df.groupBy("risk_level").count().orderBy("risk_level").show()

log.info("\n── Predicted Status Distribution ──")
pred_df.groupBy("predicted_status").count().show()

log.info("TASK 371 complete.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 372 — Dashboard note
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("TASK 372: Dashboard")
log.info("=" * 60)
log.info("All outputs written. Start dashboard with:")
log.info("    streamlit run dashboard.py")
log.info("=" * 60)

spark.stop()
log.info("Done. Spark session stopped.")
